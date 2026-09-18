import requests
import struct
import os
import zlib


# ============================================================
# SETTINGS
# ============================================================

PART1_URL = (
    "https://huggingface.co/datasets/"
    "YongchengYAO/MAMA-MIA-Lite/resolve/main/"
    "data-part001.zip"
)

PART2_URL = (
    "https://huggingface.co/datasets/"
    "YongchengYAO/MAMA-MIA-Lite/resolve/main/"
    "data-part002.zip"
)

OUTPUT_DIR = "MAMA_MIA_100"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


# ============================================================
# STEP 1 — READ ZIP CENTRAL DIRECTORY
# ============================================================

def read_zip_entries(url):

    print()
    print("Reading ZIP metadata...")
    print(url.split("/")[-1])

    # Download last 10 MB
    response = requests.get(
        url,
        headers={"Range": "bytes=-10485760"},
        timeout=120
    )

    response.raise_for_status()

    tail = response.content

    # Find ZIP64 End of Central Directory
    pos = tail.rfind(b"PK\x06\x06")

    if pos == -1:
        raise RuntimeError(
            "ZIP64 End of Central Directory not found."
        )

    # ZIP64 central directory size
    cd_size = struct.unpack_from(
        "<Q",
        tail,
        pos + 40
    )[0]

    # ZIP64 central directory offset
    cd_offset = struct.unpack_from(
        "<Q",
        tail,
        pos + 48
    )[0]

    print(
        f"Central directory size: "
        f"{cd_size / 1024 / 1024:.2f} MB"
    )

    print(
        f"Central directory offset: "
        f"{cd_offset / 1024 / 1024 / 1024:.2f} GB"
    )

    # Download central directory
    response = requests.get(
        url,
        headers={
            "Range": (
                f"bytes={cd_offset}-"
                f"{cd_offset + cd_size - 1}"
            )
        },
        timeout=120
    )

    response.raise_for_status()

    central_directory = response.content

    print(
        "Central directory downloaded:",
        f"{len(central_directory) / 1024 / 1024:.2f} MB"
    )

    # --------------------------------------------------------
    # Parse entries
    # --------------------------------------------------------

    entries = {}

    p = 0

    while p + 46 <= len(central_directory):

        if central_directory[p:p + 4] != b"PK\x01\x02":
            break

        compression = struct.unpack_from(
            "<H",
            central_directory,
            p + 10
        )[0]

        compressed_size_32 = struct.unpack_from(
            "<I",
            central_directory,
            p + 20
        )[0]

        uncompressed_size_32 = struct.unpack_from(
            "<I",
            central_directory,
            p + 24
        )[0]

        filename_length = struct.unpack_from(
            "<H",
            central_directory,
            p + 28
        )[0]

        extra_length = struct.unpack_from(
            "<H",
            central_directory,
            p + 30
        )[0]

        comment_length = struct.unpack_from(
            "<H",
            central_directory,
            p + 32
        )[0]

        local_offset_32 = struct.unpack_from(
            "<I",
            central_directory,
            p + 42
        )[0]

        filename_start = p + 46

        filename_end = (
            filename_start + filename_length
        )

        filename = central_directory[
            filename_start:filename_end
        ].decode(
            "utf-8",
            errors="replace"
        )

        extra_start = filename_end

        extra_end = (
            extra_start + extra_length
        )

        extra = central_directory[
            extra_start:extra_end
        ]

        # ----------------------------------------------------
        # ZIP64 extra field
        # ----------------------------------------------------

        compressed_size = compressed_size_32
        uncompressed_size = uncompressed_size_32
        local_offset = local_offset_32

        if (
            compressed_size_32 == 0xFFFFFFFF
            or uncompressed_size_32 == 0xFFFFFFFF
            or local_offset_32 == 0xFFFFFFFF
        ):

            ep = 0

            while ep + 4 <= len(extra):

                header_id, data_size = struct.unpack_from(
                    "<HH",
                    extra,
                    ep
                )

                data_start = ep + 4
                data_end = data_start + data_size

                if header_id == 0x0001:

                    q = data_start

                    if compressed_size_32 == 0xFFFFFFFF:

                        compressed_size = struct.unpack_from(
                            "<Q",
                            extra,
                            q
                        )[0]

                        q += 8

                    if uncompressed_size_32 == 0xFFFFFFFF:

                        uncompressed_size = struct.unpack_from(
                            "<Q",
                            extra,
                            q
                        )[0]

                        q += 8

                    if local_offset_32 == 0xFFFFFFFF:

                        local_offset = struct.unpack_from(
                            "<Q",
                            extra,
                            q
                        )[0]

                        q += 8

                    break

                ep = data_end

        entries[filename] = {
            "url": url,
            "compression": compression,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        }

        p += (
            46
            + filename_length
            + extra_length
            + comment_length
        )

    print(
        "ZIP entries found:",
        len(entries)
    )

    return entries


# ============================================================
# STEP 2 — READ BOTH ZIP FILES
# ============================================================

entries_part1 = read_zip_entries(PART1_URL)

entries_part2 = read_zip_entries(PART2_URL)


# ============================================================
# STEP 3 — COMBINE ZIP ENTRIES
# ============================================================

entries = {}

entries.update(entries_part1)
entries.update(entries_part2)

print()
print("=" * 60)
print("TOTAL ZIP ENTRIES")
print("=" * 60)

print("Part 1:", len(entries_part1))
print("Part 2:", len(entries_part2))
print("Combined:", len(entries))


# ============================================================
# STEP 4 — FIND ALL IMAGES AND MASKS
# ============================================================

images = {
    name.split("/")[-1].replace(".nii.gz", "")
    for name in entries
    if name.startswith("Images/")
}

masks = {
    name.split("/")[-1].replace(".nii.gz", "")
    for name in entries
    if name.startswith("Masks/")
}

paired = sorted(images & masks)


print()
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Total images:", len(images))
print("Total masks :", len(masks))
print("Paired cases:", len(paired))


# ============================================================
# STEP 5 — SEPARATE COHORTS
# ============================================================

ispy1 = sorted(
    case for case in paired
    if case.startswith("ISPY1_")
)

ispy2 = sorted(
    case for case in paired
    if case.startswith("ISPY2_")
)

nact = sorted(
    case for case in paired
    if case.startswith("NACT_")
)

duke = sorted(
    case for case in paired
    if case.startswith("DUKE_")
)


print()
print("=" * 60)
print("COHORT DISTRIBUTION")
print("=" * 60)

print("ISPY1:", len(ispy1))
print("ISPY2:", len(ispy2))
print("NACT :", len(nact))
print("DUKE :", len(duke))


# ============================================================
# STEP 6 — SELECT EXACTLY 100 CASES
# ============================================================

selected_ispy1 = ispy1[:25]

selected_ispy2 = ispy2[:25]

selected_nact = nact[:25]

selected_duke = duke[:25]


# Safety checks
if len(selected_ispy1) != 25:
    raise RuntimeError(
        "Could not select 25 ISPY1 cases."
    )

if len(selected_ispy2) != 25:
    raise RuntimeError(
        "Could not select 25 ISPY2 cases."
    )

if len(selected_nact) != 25:
    raise RuntimeError(
        "Could not select 25 NACT cases."
    )

if len(selected_duke) != 25:
    raise RuntimeError(
        "Could not select 25 Duke cases."
    )


selected = (
    selected_ispy1
    + selected_ispy2
    + selected_nact
    + selected_duke
)


if len(selected) != 100:
    raise RuntimeError(
        f"Expected 100 cases, "
        f"but selected {len(selected)}."
    )


print()
print("=" * 60)
print("SELECTED CASES")
print("=" * 60)

print("ISPY1:", len(selected_ispy1))
print("ISPY2:", len(selected_ispy2))
print("NACT :", len(selected_nact))
print("DUKE :", len(selected_duke))
print("TOTAL:", len(selected))


# ============================================================
# STEP 7 — DOWNLOAD ONE ZIP ENTRY
# ============================================================

def download_entry(filename):

    info = entries[filename]

    url = info["url"]

    local_offset = info["local_offset"]

    compressed_size = info["compressed_size"]


    # --------------------------------------------------------
    # Download local ZIP header
    # --------------------------------------------------------

    header_response = requests.get(
        url,
        headers={
            "Range": (
                f"bytes={local_offset}-"
                f"{local_offset + 29}"
            )
        },
        timeout=120
    )

    header_response.raise_for_status()

    header = header_response.content


    if header[:4] != b"PK\x03\x04":

        raise RuntimeError(
            f"\nInvalid local ZIP header!\n"
            f"File: {filename}\n"
            f"ZIP offset: {local_offset}\n"
            f"URL: {url}"
        )


    # Local ZIP header
    filename_length = struct.unpack_from(
        "<H",
        header,
        26
    )[0]

    extra_length = struct.unpack_from(
        "<H",
        header,
        28
    )[0]


    # Actual compressed data location
    data_start = (
        local_offset
        + 30
        + filename_length
        + extra_length
    )

    data_end = (
        data_start
        + compressed_size
        - 1
    )


    print(
        f"  Downloading {filename}"
    )

    print(
        f"  Size: "
        f"{compressed_size / 1024 / 1024:.2f} MB"
    )


    # --------------------------------------------------------
    # Download compressed bytes
    # --------------------------------------------------------

    response = requests.get(
        url,
        headers={
            "Range": (
                f"bytes={data_start}-{data_end}"
            )
        },
        timeout=600
    )

    response.raise_for_status()

    compressed_data = response.content


    if len(compressed_data) != compressed_size:

        raise RuntimeError(
            f"\nIncomplete download!\n"
            f"File: {filename}\n"
            f"Expected: {compressed_size} bytes\n"
            f"Received: {len(compressed_data)} bytes"
        )


    return compressed_data


# ============================================================
# STEP 8 — DECOMPRESS ENTRY
# ============================================================

def extract_entry(filename):

    info = entries[filename]

    compression = info["compression"]

    compressed_data = download_entry(filename)


    # ZIP compression method 0
    if compression == 0:

        data = compressed_data


    # ZIP compression method 8 = DEFLATE
    elif compression == 8:

        try:

            data = zlib.decompress(
                compressed_data,
                -15
            )

        except zlib.error as e:

            raise RuntimeError(
                f"DEFLATE decompression failed "
                f"for {filename}: {e}"
            )


    else:

        raise RuntimeError(
            f"Unsupported compression method "
            f"{compression} for {filename}"
        )


    return data


# ============================================================
# STEP 9 — DOWNLOAD 100 IMAGE-MASK PAIRS
# ============================================================

for index, case in enumerate(selected, 1):

    print()
    print("=" * 60)
    print(
        f"CASE {index}/100: {case}"
    )
    print("=" * 60)


    image_name = (
        f"Images/{case}.nii.gz"
    )

    mask_name = (
        f"Masks/{case}.nii.gz"
    )


    image_path = os.path.join(
        IMAGE_DIR,
        f"{case}.nii.gz"
    )

    mask_path = os.path.join(
        MASK_DIR,
        f"{case}.nii.gz"
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if os.path.exists(image_path):

        print(
            "  Image already exists — skipping."
        )

    else:

        image_data = extract_entry(
            image_name
        )

        with open(
            image_path,
            "wb"
        ) as f:

            f.write(image_data)

        print(
            "  Image saved."
        )


    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    if os.path.exists(mask_path):

        print(
            "  Mask already exists — skipping."
        )

    else:

        mask_data = extract_entry(
            mask_name
        )

        with open(
            mask_path,
            "wb"
        ) as f:

            f.write(mask_data)

        print(
            "  Mask saved."
        )


    print(
        f"  Completed: {case}"
    )


# ============================================================
# STEP 10 — FINAL VERIFICATION
# ============================================================

image_files = {
    filename.replace(
        ".nii.gz",
        ""
    )
    for filename in os.listdir(IMAGE_DIR)
    if filename.endswith(".nii.gz")
}


mask_files = {
    filename.replace(
        ".nii.gz",
        ""
    )
    for filename in os.listdir(MASK_DIR)
    if filename.endswith(".nii.gz")
}


paired_final = (
    image_files & mask_files
)


print()
print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

print(
    "Images saved :",
    len(image_files)
)

print(
    "Masks saved  :",
    len(mask_files)
)

print(
    "Paired cases :",
    len(paired_final)
)


if len(paired_final) == 100:

    print()
    print(
        "SUCCESS: "
        "100 complete image-mask pairs are ready."
    )

else:

    print()
    print(
        "WARNING: Expected 100 complete pairs, "
        f"but found {len(paired_final)}."
    )


# ============================================================
# STEP 11 — FINAL COHORT LIST
# ============================================================

print()
print("=" * 60)
print("SELECTED COHORT COUNTS")
print("=" * 60)

final_ispy1 = [
    x for x in paired_final
    if x.startswith("ISPY1_")
]

final_ispy2 = [
    x for x in paired_final
    if x.startswith("ISPY2_")
]

final_nact = [
    x for x in paired_final
    if x.startswith("NACT_")
]

final_duke = [
    x for x in paired_final
    if x.startswith("DUKE_")
]


print(
    "ISPY1:",
    len(final_ispy1)
)

print(
    "ISPY2:",
    len(final_ispy2)
)

print(
    "NACT :",
    len(final_nact)
)

print(
    "DUKE :",
    len(final_duke)
)

print()
print("Done.")