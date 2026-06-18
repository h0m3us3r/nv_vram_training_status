// nv_vram_training_status — per-IC VRAM DQ-training fault locator for NVIDIA GPUs
//                           (Pascal / Turing / Ampere / Ada; GDDR5/5X/6/6X)
//
// When a GPU's VRAM has a dead or marginal memory IC, the memory controller
// cannot *train* the data (DQ) lines for that device — the board fails to bring
// up its frame buffer ("Memory DQ write training failed") and falls back to a
// low clock or won't initialize at all. The controller latches the
// per-subpartition training result in a frame-buffer-partition (FBPA) status
// register.
//
// This tool reads that register read-only over BAR0 MMIO and reports, per
// partition and subpartition, whether DQ training finished cleanly or errored
// out — localizing the fault to one 16-bit half of a memory channel.
//
// Access is a pure read-only mmap of the GPU's PCI BAR0 via the sysfs
// `resource0` file: no register writes, no VRAM access, no pattern test. It only
// reads status words the hardware already populated, so it is safe to run on a
// board whose memory is too broken to train — an active write/read memory test
// would need a working, trained frame buffer just to start.
//
// Scope
// -----
//   Supported: any GDDR5/GDDR5X/GDDR6/GDDR6X NVIDIA GPU from Pascal through Ada
//   (GTX 10xx, GTX 16xx, RTX 20/30/40, and the matching pro/datacenter parts).
//   Not supported: HBM parts (P100, V100, A100, H100) — different FB layout, and
//   their on-package memory is not board-repairable anyway. Pre-Pascal
//   (Maxwell/Kepler) do not expose this register at this offset.
//
// Chip detection
// --------------
//   The chip is identified from NV_PMC_BOOT_0 (BAR0 + 0x0):
//       chip_id = (boot0 >> 20) & 0x1ff   // 0x132=GP102, 0x162=TU102, 0x174=GA102, 0x192=AD102
//       arch    = chip_id >> 4            // 0x13=Pascal 0x16=Turing 0x17=Ampere 0x19=Ada
//
// Register layout (FBPA block)
// ----------------------------
//     base   = 0x900000              // FBPA (memory-controller) register block
//     stride = 0x4000                // per memory partition
//     status(part) = base + part*stride + 0x974
//
//   The status word packs two 2-bit fields, one per subpartition:
//       subpartition 0 = bits[1:0]
//       subpartition 1 = bits[3:2]
//   Field value -> meaning:
//       0 = finished            (training completed cleanly — OK)
//       1 = in progress         (still training — transient)
//       2 = errored out         (training FAILED)
//       3 = done with errors    (training FAILED)
//
//   The register address (0x900974, partition 0), the 0x974 offset and the
//   2-bit decode are consistent across Pascal, Turing, Ampere and Ada — the FBPA
//   register block has been stable across these generations.
//
//   A subpartition is one 16-bit half of a partition's 32-bit memory channel.
//   Translating (partition, subpartition) to a physical IC position on the board
//   is intentionally out of scope here (planned as a separate, board-specific
//   location view).
//
//   Partitions that do not exist on this board read back the GPU PRI "bad read"
//   sentinel 0xBADF____ and are skipped.
//
// The 0x974 offset and the field encoding are not from a published NVIDIA
// hardware reference. Validate against a card with a known-bad IC before
// trusting a FAIL verdict for a repair decision (see --raw).
//
// Error counts (--errors)
// -----------------------
//   Each partition also exposes free-running per-subpartition error counters at
//   FBPA offset 0x480..0x48c (four 32-bit words). These are read read-only and
//   accumulate detected/corrected data-bus (EDC/ECC) errors. A partition that
//   trained cleanly but is steadily logging errors points to a *marginal* IC —
//   one that is intermittent, or only fails under load or temperature — which a
//   pass/fail training verdict alone will not catch. Reading them writes nothing,
//   so --errors keeps the same read-only safety as the default mode. The four
//   words pair up by subpartition — subp0 = 0x480+0x484, subp1 = 0x488+0x48c,
//   matching the controller's own per-subpartition counter-select grouping; the
//   two words in a pair are the subpartition's byte-lanes (see --raw).
//
// Forced retrain (--retrain)   [THE ONLY MODE THAT WRITES TO THE GPU]
// -------------------------------------------------------------------
//   --retrain asks the memory controller to re-run DQ training by asserting the
//   request bit (0x80000000) in the two broadcast training-command registers
//   (FBPA 0x910 / 0x914), waits for the per-subpartition status to settle, then
//   prints the fresh result. Use it to tell an intermittent fault from a hard
//   one, or to re-validate after a reflow/reball/IC swap without a full reboot.
//
//   This is the only path that writes to the GPU; it needs R/W BAR access
//   (the BAR is opened O_RDWR only in this mode) and an explicit confirmation
//   (or --yes). It is UNSAFE on a card whose framebuffer is in active use —
//   re-training the live memory link can freeze the display or hang the
//   machine. Run it on a bench card driven from a second GPU / serial console.
//   The trigger writes only the request bit on top of the controller's existing
//   command words (a minimal kick, not a full vendor training sequence) and the
//   offsets are not from a published reference: treat this mode as experimental.
//
// Build:  cc -O2 -o nv_vram_training_status nv_vram_training_status.c
// Run:    sudo ./nv_vram_training_status [--raw] [--color] [--errors]
//         sudo ./nv_vram_training_status --retrain [--yes]   (writes to the GPU)
//
// If the BAR mmap fails (EPERM via /dev/mem, or EINVAL via the sysfs resource),
// the kernel's CONFIG_IO_STRICT_DEVMEM is refusing access to the driver-owned
// BAR. Boot with the kernel parameter `iomem=relaxed`; Secure Boot / kernel
// lockdown must also be off.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>

#define PMC_BOOT_0    0x000000u    // NV_PMC_BOOT_0 — chip identification
#define FBPA_BASE     0x900000u    // FBPA (memory-controller) register block
#define FBPA_STRIDE   0x4000u      // per memory partition
#define TRAIN_OFF     0x0974u      // NV_PFB_FBPA_TRAINING_STATUS (per partition)
#define ECC_OFF       0x0480u      // per-partition error counters: 4 words at 0x480..0x48c
#define ECC_NREG      4
#define TRAIN_CMD0    0x9a0910u    // broadcast (all-partition) training-command registers
#define TRAIN_CMD1    0x9a0914u
#define TRAIN_CMD_GO  0x80000000u  // request bit: asserting it kicks a training run
#define NUM_FBPA      24           // scan up to 24; non-existent ones are skipped
#define NUM_SUBP      2            // subpartitions per partition
#define MAP_LEN       0x1000000u   // map enough of BAR0 to cover the FBPA block

// SKU names for the header line only — detection comes from NV_PMC_BOOT_0, so an
// unlisted card of a supported architecture still works. Edit/extend freely.
typedef struct { uint16_t dev_id; const char *name; const char *vram; } board_t;

static const board_t BOARDS[] = {
    // Ada (AD10x)
    { 0x2684, "RTX 4090",          "GDDR6X" },
    { 0x2685, "RTX 4090 D",        "GDDR6X" },
    { 0x2702, "RTX 4080 Super",    "GDDR6X" },
    { 0x2704, "RTX 4080",          "GDDR6X" },
    { 0x2705, "RTX 4070 Ti Super", "GDDR6X" },
    { 0x2782, "RTX 4070 Ti",       "GDDR6X" },
    { 0x2783, "RTX 4070 Super",    "GDDR6X" },
    { 0x2786, "RTX 4070",          "GDDR6X" },
    { 0x2860, "RTX 4070 Mobile",   "GDDR6"  },
    { 0x26b1, "RTX A6000 (Ada)",   "GDDR6"  },
    { 0x26b9, "L40S",              "GDDR6"  },
    { 0x27b8, "L4",                "GDDR6"  },
    // Ampere (GA10x)
    { 0x2203, "RTX 3090 Ti",       "GDDR6X" },
    { 0x2204, "RTX 3090",          "GDDR6X" },
    { 0x2208, "RTX 3080 Ti",       "GDDR6X" },
    { 0x2206, "RTX 3080",          "GDDR6X" },
    { 0x2216, "RTX 3080 LHR",      "GDDR6X" },
    { 0x2231, "RTX A5000",         "GDDR6"  },
    { 0x2232, "RTX A4500",         "GDDR6"  },
    { 0x2236, "A10",               "GDDR6"  },
    { 0x2482, "RTX 3070 Ti",       "GDDR6X" },
    { 0x2484, "RTX 3070",          "GDDR6"  },
    { 0x2488, "RTX 3070 LHR",      "GDDR6"  },
    { 0x2486, "RTX 3060 Ti",       "GDDR6"  },
    { 0x2503, "RTX 3060",          "GDDR6"  },
    { 0x2504, "RTX 3060 LHR",      "GDDR6"  },
    { 0x2531, "RTX A2000",         "GDDR6"  },
    { 0x2571, "RTX A2000",         "GDDR6"  },
    // Turing (TU10x/11x)
    { 0x1e02, "TITAN RTX",         "GDDR6"  },
    { 0x1e04, "RTX 2080 Ti",       "GDDR6"  },
    { 0x1e07, "RTX 2080 Ti",       "GDDR6"  },
    { 0x1e81, "RTX 2080 Super",    "GDDR6"  },
    { 0x1e82, "RTX 2080",          "GDDR6"  },
    { 0x1e84, "RTX 2070 Super",    "GDDR6"  },
    { 0x1f02, "RTX 2070",          "GDDR6"  },
    { 0x1f07, "RTX 2070",          "GDDR6"  },
    { 0x1f06, "RTX 2060 Super",    "GDDR6"  },
    { 0x1f08, "RTX 2060",          "GDDR6"  },
    { 0x1f15, "RTX 2060",          "GDDR6"  },
    { 0x2182, "GTX 1660 Ti",       "GDDR6"  },
    { 0x21c4, "GTX 1660 Super",    "GDDR6"  },
    { 0x2184, "GTX 1660",          "GDDR5"  },
    { 0x1f82, "GTX 1650",          "GDDR5"  },
    { 0x2188, "GTX 1650",          "GDDR6"  },
    // Pascal (GP10x) — GDDR5X / GDDR5  (GP100 is HBM and excluded)
    { 0x1b00, "TITAN X (Pascal)",  "GDDR5X" },
    { 0x1b02, "TITAN Xp",          "GDDR5X" },
    { 0x1b06, "GTX 1080 Ti",       "GDDR5X" },
    { 0x1b80, "GTX 1080",          "GDDR5X" },
    { 0x1b81, "GTX 1070",          "GDDR5"  },
    { 0x1b82, "GTX 1070 Ti",       "GDDR5"  },
    { 0x1b83, "GTX 1060 6GB",      "GDDR5"  },
    { 0x1c02, "GTX 1060 3GB",      "GDDR5"  },
    { 0x1c03, "GTX 1060 6GB",      "GDDR5"  },
    { 0x1c04, "GTX 1060",          "GDDR5"  },
    { 0x1c06, "GTX 1060 6GB",      "GDDR5"  },
    { 0x1c81, "GTX 1050",          "GDDR5"  },
    { 0x1c82, "GTX 1050 Ti",       "GDDR5"  },
    { 0x1c83, "GTX 1050 3GB",      "GDDR5"  },
    { 0x1d01, "GT 1030",           "GDDR5"  },
};

static const board_t *lookup_board(uint16_t dev_id) {
    for (size_t i = 0; i < sizeof BOARDS / sizeof BOARDS[0]; i++)
        if (BOARDS[i].dev_id == dev_id) return &BOARDS[i];
    return NULL;
}

// Architecture name from the NV_PMC_BOOT_0 chip-id high nibble.
static const char *arch_name(unsigned chip_id) {
    switch (chip_id >> 4) {
        case 0x11: return "Kepler";
        case 0x12: return "Maxwell";
        case 0x13: return "Pascal";
        case 0x14: return "Volta";
        case 0x16: return "Turing";
        case 0x17: return "Ampere";
        case 0x18: return "Hopper";
        case 0x19: return "Ada";
        default:   return "unknown";
    }
}

// On-package HBM parts: different FB layout, not addressable here (and not
// board-repairable). Identified by die — P100/V100/A100/H100.
static int chip_is_hbm(unsigned chip_id) {
    if ((chip_id >> 4) == 0x14) return 1;       // Volta (GV100)
    if ((chip_id >> 4) == 0x18) return 1;       // Hopper (GH100)
    return chip_id == 0x130     /* GP100 */
        || chip_id == 0x172;    /* GA100 */
}

// 2-bit per-subpartition training state.
enum { ST_FINISHED = 0, ST_IN_PROGRESS = 1, ST_ERRORED = 2, ST_DONE_ERR = 3 };
static const char *state_name(unsigned v) {
    switch (v & 3) {
        case ST_FINISHED:    return "ok";
        case ST_IN_PROGRESS: return "training";
        case ST_ERRORED:     return "FAIL";
        default:             return "FAIL*";
    }
}
static int state_is_fail(unsigned v) { v &= 3; return v == ST_ERRORED || v == ST_DONE_ERR; }

#define C_RESET "\033[0m"
static const char *state_color(unsigned v) {
    if (state_is_fail(v))          return "\033[1;31m";
    if ((v & 3) == ST_IN_PROGRESS) return "\033[33m";
    return "\033[32m";
}
static void print_state(unsigned v, int color) {
    if (color) printf("%s%-9s%s", state_color(v), state_name(v), C_RESET);
    else       printf("%-9s", state_name(v));
}

// A partition that does not exist on this board reads the PRI "bad read"
// sentinel 0xBADF____ (or all-ones) in its training-status word.
static int part_present(volatile uint8_t *map, int part) {
    uint32_t off = FBPA_BASE + (uint32_t)part * FBPA_STRIDE + TRAIN_OFF;
    uint32_t reg = *(volatile uint32_t *)(map + off);
    return !((reg >> 16) == 0xbadf || reg == 0xffffffff);
}

// Read-only per-partition error-count report. Returns the number of partitions
// with a nonzero count (a marginal-IC indicator even when training passed).
static int report_errors(volatile uint8_t *map, int color, int raw) {
    printf("  FBPA  subp0-errs  subp1-errs%s\n", raw ? "  raw 480..48c" : "");
    printf("  ----  ----------  ----------%s\n", raw ? "  ------------" : "");
    int populated = 0, flagged = 0;
    for (int part = 0; part < NUM_FBPA; part++) {
        if (!part_present(map, part)) continue;
        populated++;
        uint32_t pbase = FBPA_BASE + (uint32_t)part * FBPA_STRIDE;
        uint32_t c[ECC_NREG];
        for (int i = 0; i < ECC_NREG; i++)
            c[i] = *(volatile uint32_t *)(map + pbase + ECC_OFF + (uint32_t)i * 4u);
        uint64_t s0 = (uint64_t)c[0] + c[1];
        uint64_t s1 = (uint64_t)c[2] + c[3];
        const char *r0 = (color && s0) ? "\033[1;31m" : "";
        const char *r1 = (color && s1) ? "\033[1;31m" : "";
        printf("  %4d  %s%10llu%s  %s%10llu%s", part,
               r0, (unsigned long long)s0, *r0 ? C_RESET : "",
               r1, (unsigned long long)s1, *r1 ? C_RESET : "");
        if (raw) printf("  %08x %08x %08x %08x", c[0], c[1], c[2], c[3]);
        printf("\n");
        if (s0 || s1) flagged++;
    }
    return populated ? flagged : -1;
}

// EXPERIMENTAL, WRITES TO THE GPU. Asserts the request bit in the broadcast
// training-command registers to re-run DQ training, then waits (up to ~2s) for
// every present subpartition to leave the "in progress" state.
static void do_retrain(volatile uint8_t *map) {
    uint32_t c0 = *(volatile uint32_t *)(map + TRAIN_CMD0);
    uint32_t c1 = *(volatile uint32_t *)(map + TRAIN_CMD1);
    *(volatile uint32_t *)(map + TRAIN_CMD0) = c0 | TRAIN_CMD_GO;
    *(volatile uint32_t *)(map + TRAIN_CMD1) = c1 | TRAIN_CMD_GO;

    for (int t = 0; t < 200; t++) {
        usleep(10000);                 // 10 ms
        int busy = 0;
        for (int part = 0; part < NUM_FBPA; part++) {
            if (!part_present(map, part)) continue;
            uint32_t off = FBPA_BASE + (uint32_t)part * FBPA_STRIDE + TRAIN_OFF;
            uint32_t r = *(volatile uint32_t *)(map + off);
            if ((r & 3) == ST_IN_PROGRESS || ((r >> 2) & 3) == ST_IN_PROGRESS) busy = 1;
        }
        if (!busy) return;
    }
}

// Find the first NVIDIA GPU via sysfs: fills res0_path, pci_id, dev_id, bar0_len;
// returns the BAR0 physical base (0 on failure).
static uint64_t find_gpu(char *res0_path, size_t res0_sz, char *pci_id, size_t pci_sz,
                         uint16_t *dev_id, uint64_t *bar0_len) {
    DIR *d = opendir("/sys/bus/pci/devices");
    if (!d) { perror("opendir /sys/bus/pci/devices"); return 0; }
    struct dirent *e;
    char path[512], buf[256];
    uint64_t bar0 = 0;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/vendor", e->d_name);
        FILE *f = fopen(path, "r"); if (!f) continue;
        buf[0] = 0; if (!fgets(buf, sizeof buf, f)) { fclose(f); continue; } fclose(f);
        if (strncmp(buf, "0x10de", 6) != 0) continue;
        snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/class", e->d_name);
        f = fopen(path, "r"); if (!f) continue;
        buf[0] = 0; if (!fgets(buf, sizeof buf, f)) { fclose(f); continue; } fclose(f);
        if (strncmp(buf, "0x0300", 6) != 0 && strncmp(buf, "0x0302", 6) != 0) continue;
        unsigned did = 0;
        snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/device", e->d_name);
        f = fopen(path, "r");
        if (f) { if (fscanf(f, "%x", &did) != 1) did = 0; fclose(f); }
        snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource", e->d_name);
        f = fopen(path, "r"); if (!f) continue;
        unsigned long long start = 0, end = 0, flags = 0;
        if (fscanf(f, "%llx %llx %llx", &start, &end, &flags) == 3 && end >= start) {
            bar0 = start;
            *bar0_len = end - start + 1;
        }
        fclose(f);
        if (bar0) {
            *dev_id = (uint16_t)did;
            snprintf(pci_id, pci_sz, "%.*s", (int)(pci_sz - 1), e->d_name);
            snprintf(res0_path, res0_sz, "/sys/bus/pci/devices/%s/resource0", e->d_name);
            break;
        }
    }
    closedir(d);
    return bar0;
}

int main(int argc, char **argv) {
    int color = 0, raw = 0, errors = 0, retrain = 0, assume_yes = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-c") || !strcmp(argv[a], "--color")) color = 1;
        else if (!strcmp(argv[a], "-r") || !strcmp(argv[a], "--raw")) raw = 1;
        else if (!strcmp(argv[a], "-e") || !strcmp(argv[a], "--errors")) errors = 1;
        else if (!strcmp(argv[a], "--retrain")) retrain = 1;
        else if (!strcmp(argv[a], "-y") || !strcmp(argv[a], "--yes")) assume_yes = 1;
        else if (!strcmp(argv[a], "-h") || !strcmp(argv[a], "--help")) {
            printf("usage: %s [--raw] [--color] [--errors]\n"
                   "       %s --retrain [--yes] [--color]\n"
                   "  Reports NVIDIA VRAM per-subpartition DQ-training status (read-only).\n"
                   "  Works on Pascal/Turing/Ampere/Ada (GTX 10xx/16xx, RTX 20/30/40). HBM unsupported.\n"
                   "  --raw,     -r   also print the raw register word(s) per partition\n"
                   "  --color,   -c   green = ok, red = failed\n"
                   "  --errors,  -e   also report per-subpartition error counts (read-only)\n"
                   "  --retrain       re-run DQ training first, then report  [WRITES TO THE GPU]\n"
                   "  --yes,     -y   skip the --retrain confirmation prompt\n"
                   "  exit: 0 all trained, 3 a subpartition failed, 1 cannot read, 2 unsupported\n",
                   argv[0], argv[0]);
            return 0;
        } else { fprintf(stderr, "unknown argument: %s (try --help)\n", argv[a]); return 2; }
    }
    if (assume_yes && !retrain)
        fprintf(stderr, "note: --yes only applies to --retrain; ignoring.\n");

    char res0_path[512], pci_id[64] = "";
    uint16_t dev_id = 0;
    uint64_t bar0_len = 0;
    uint64_t bar0 = find_gpu(res0_path, sizeof res0_path, pci_id, sizeof pci_id,
                             &dev_id, &bar0_len);
    if (!bar0) { fprintf(stderr, "No NVIDIA GPU found on the PCI bus.\n"); return 1; }
    if (bar0_len < MAP_LEN) {
        fprintf(stderr, "BAR0 too small (0x%llx) for the FBPA register block.\n",
                (unsigned long long)bar0_len);
        return 1;
    }

    int fd = open(res0_path, retrain ? O_RDWR : O_RDONLY);
    if (fd < 0) { perror("open resource0 (run as root)"); return 1; }
    int prot = retrain ? (PROT_READ | PROT_WRITE) : PROT_READ;
    volatile uint8_t *map = mmap(NULL, MAP_LEN, prot, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap resource0");
        fprintf(stderr, "EPERM  => kernel lockdown (Secure Boot): disable it.\n"
                        "EINVAL => CONFIG_IO_STRICT_DEVMEM: boot with iomem=relaxed.\n");
        close(fd); return 1;
    }

    // Identify the chip from NV_PMC_BOOT_0 (authoritative, read-only).
    uint32_t boot0 = *(volatile uint32_t *)(map + PMC_BOOT_0);
    unsigned chip_id = (boot0 >> 20) & 0x1ff;
    unsigned arch = chip_id >> 4;
    const board_t *b = lookup_board(dev_id);

    printf("VRAM DQ-training status   GPU %s  [%04x]  %s %s   %s (chip 0x%03x)   BAR0 0x%llx\n\n",
           pci_id, dev_id, b ? b->name : "(unlisted SKU)", b ? b->vram : "",
           arch_name(chip_id), chip_id, (unsigned long long)bar0);

    if (boot0 != 0xffffffff) {
        if (chip_is_hbm(chip_id)) {
            fprintf(stderr,
                "This is an HBM part (%s) — its on-package memory uses a different FB\n"
                "layout this tool does not read, and is not board-repairable. Unsupported.\n",
                arch_name(chip_id));
            munmap((void *)map, MAP_LEN); close(fd); return 2;
        }
        if (arch < 0x13) {
            fprintf(stderr,
                "This chip (%s) predates Pascal; the FBPA DQ-training-status register is\n"
                "not present at this offset on Maxwell/Kepler. Unsupported.\n", arch_name(chip_id));
            munmap((void *)map, MAP_LEN); close(fd); return 2;
        }
        if (arch > 0x19)
            fprintf(stderr,
                "Note: %s is newer than Ada and untested; assuming the same register layout.\n\n",
                arch_name(chip_id));
    }

    if (retrain) {
        if (!assume_yes) {
            fprintf(stderr,
                "WARNING: --retrain WRITES to the GPU and re-runs the live memory\n"
                "training. On a card whose framebuffer is in use this can freeze the\n"
                "display or hang the machine. Only proceed on a bench/secondary GPU.\n"
                "Type 'yes' to continue: ");
            char line[16] = {0};
            if (!fgets(line, sizeof line, stdin) || strncmp(line, "yes", 3) != 0) {
                fprintf(stderr, "Aborted.\n");
                munmap((void *)map, MAP_LEN); close(fd); return 1;
            }
        }
        printf("Re-running DQ training...\n");
        do_retrain(map);
        printf("\n");
    }

    printf("  FBPA  subp0      subp1%s\n", raw ? "        raw" : "");
    printf("  ----  ---------  ---------%s\n", raw ? "  ----------" : "");

    int populated = 0, failures = 0;
    int fail_fbpa[NUM_FBPA * NUM_SUBP], fail_subp[NUM_FBPA * NUM_SUBP];
    for (int part = 0; part < NUM_FBPA; part++) {
        uint32_t off = FBPA_BASE + (uint32_t)part * FBPA_STRIDE + TRAIN_OFF;
        uint32_t reg = *(volatile uint32_t *)(map + off);
        if ((reg >> 16) == 0xbadf || reg == 0xffffffff) continue;  // absent partition
        populated++;

        unsigned st[NUM_SUBP] = { reg & 3, (reg >> 2) & 3 };
        printf("  %4d  ", part);
        print_state(st[0], color); printf("  ");
        print_state(st[1], color);
        if (raw) printf("  0x%08x", reg);
        printf("\n");

        for (int s = 0; s < NUM_SUBP; s++)
            if (state_is_fail(st[s])) {
                fail_fbpa[failures] = part; fail_subp[failures] = s; failures++;
            }
    }

    printf("\n");
    if (!populated) {
        fprintf(stderr,
            "No memory partitions responded — unexpected for %s.\n"
            "The FBPA register offset may differ on this board.\n", arch_name(chip_id));
        munmap((void *)map, MAP_LEN); close(fd);
        return 1;
    }

    if (failures == 0) {
        printf("PASS: %d partitions present, all subpartitions trained cleanly.\n", populated);
    } else {
        printf("FAIL: %d partitions present, %d subpartition(s) failed DQ training:\n",
               populated, failures);
        for (int i = 0; i < failures; i++)
            printf("    FBPA %d, subpartition %d\n", fail_fbpa[i], fail_subp[i]);
    }

    if (errors) {
        printf("\nError counts (read-only):\n");
        int flagged = report_errors(map, color, raw);
        if (flagged > 0)
            printf("\n%d partition(s) report nonzero errors — suspect a marginal IC there\n"
                   "even where DQ training passed.\n", flagged);
        else if (flagged == 0)
            printf("\nAll present partitions report zero accumulated errors.\n");
    }

    munmap((void *)map, MAP_LEN);
    close(fd);
    return failures ? 3 : 0;
}
