# nv_vram_training_status — per-IC VRAM DQ-training fault locator

NVIDIA Pascal / Turing / Ampere / Ada — GDDR5 / GDDR5X / GDDR6 / GDDR6X
(GTX 10xx, GTX 16xx, RTX 20 / 30 / 40 series and the matching pro/datacenter parts)

When a GPU's VRAM has a dead or marginal memory IC, the memory controller can't
**train** the data (DQ) lines for that device. The board fails to bring up its
frame buffer — you see *"Memory DQ write training failed"*, and the card falls
back to a low clock or won't initialize. The controller latches the
per-subpartition training result in a frame-buffer-partition (FBPA) status
register.

This tool reads that register **read-only over BAR0 MMIO** and reports, per
partition and subpartition, whether DQ training finished cleanly or errored out —
localizing the fault to one 16-bit half of a memory channel.

Access is a pure read-only `mmap` of the GPU's PCI BAR0 via sysfs `resource0`:
**no register writes, no VRAM access, no pattern test.** It only reads status
words the hardware already populated, so it is safe to run on a board whose
memory is too broken to train. (An active write/read memory test can't help here
— it needs a working, trained frame buffer just to start, which is exactly what a
bad IC prevents.)

## What it reports

```
VRAM DQ-training status   GPU 0000:16:00.0  [1b06]  GTX 1080 Ti GDDR5X   Pascal (chip 0x132)   BAR0 0xf6000000

  FBPA  subp0      subp1
  ----  ---------  ---------
     0  ok         ok
     1  ok         FAIL
     ...

FAIL: 6 partitions present, 1 subpartition(s) failed DQ training:
    FBPA 1, subpartition 1
```

Exit status: `0` = all trained, `3` = at least one subpartition failed, `1` =
could not read, `2` = unsupported chip (HBM / pre-Pascal).

## Chip detection & compatibility

The chip is identified directly from the GPU via **`NV_PMC_BOOT_0`** (BAR0+0):
`chip_id = (boot0 >> 20) & 0x1ff`, `arch = chip_id >> 4`. The SKU name comes from
the PCI device id (the `BOARDS[]` table). Because detection is register-based, an
unlisted card of a supported architecture **still works** — the partition scan
discovers the populated channels regardless.

| Arch | chip nibble | Examples |
|------|-------------|----------|
| Pascal | `0x13` | GTX 1050–1080 Ti, TITAN X/Xp (GDDR5/5X) |
| Turing | `0x16` | GTX 1650/1660, RTX 2060–2080 Ti, TITAN RTX |
| Ampere | `0x17` | RTX 3060–3090 Ti, A2000/A4500/A5000, A10 |
| Ada    | `0x19` | RTX 4070–4090, L4, L40S, RTX A6000 (Ada) |

The FBPA training-status register (base `0x900000`, stride `0x4000`, offset
`0x974`) and its 2-bit-per-subpartition encoding are consistent across these
architectures. Confirm against your own hardware before relying on a result (see
*Validating the register decode* below).

**Not supported:**
- **HBM parts** (P100, V100, A100, H100) — different FB layout, and the
  on-package memory isn't board-repairable anyway. Reported and skipped.
- **Pre-Pascal** (Maxwell, Kepler) — this register is not present at this offset.

Architectures newer than Ada are attempted on a best-effort basis (the register
family has been stable for many generations) with a note.

## How it works

```
base   = 0x900000        # FBPA (memory-controller) register block
stride = 0x4000          # per memory partition

training_status(part) = base + part*stride + 0x974

Packs two 2-bit fields, one per subpartition:
    subpartition 0 = bits[1:0]
    subpartition 1 = bits[3:2]

field value -> meaning
    0 = finished           (trained cleanly — "ok")
    1 = in progress        (transient — "training")
    2 = errored out        (training FAILED)
    3 = done with errors   (training FAILED)
```

A **subpartition** is one 16-bit half of a partition's 32-bit memory channel. The
tool reports the failing `(partition, subpartition)` directly. Translating that to
a physical IC silkscreen position is board-specific and intentionally out of
scope here — a location-based view is planned as a separate feature.

Partitions that don't exist on the board read back the GPU's PRI bad-read
sentinel `0xBADF____` and are skipped automatically.

## Build & run

```
cc -O2 -o nv_vram_training_status nv_vram_training_status.c
sudo ./nv_vram_training_status            # status table + verdict
sudo ./nv_vram_training_status --color    # green = ok, red = failed
sudo ./nv_vram_training_status --raw      # also show the raw status word per partition
```

## Requirements

The GPU BAR0 is owned by the `nvidia` driver, so by default the kernel
(`CONFIG_IO_STRICT_DEVMEM`) refuses to map it from user space. If the mmap fails:

- **Boot with `iomem=relaxed`** — add it to `GRUB_CMDLINE_LINUX_DEFAULT` in
  `/etc/default/grub`, run `sudo update-grub`, and reboot.
- **Secure Boot / kernel lockdown must be off** — lockdown blocks BAR access
  outright (`mmap` returns `EPERM`).

If the memory fault prevents the `nvidia` driver from loading at all, the BAR is
unclaimed and still mappable — the tool does not need the driver bound, only the
GPU present on the PCI bus with BAR0 assigned.

## Validating the register decode

The `0x974` status offset and the 2-bit field encoding are not from a published
NVIDIA hardware reference. Before using a `FAIL` verdict for an RMA/repair
decision:

- Run on a **known-good** card — every populated subpartition should read `ok`.
- Run on a card with a **known-bad** IC — the flagged `(FBPA, subpartition)`
  should correspond to the physically failing device.
- Use `--raw` to inspect the unmasked status words if anything looks off.
