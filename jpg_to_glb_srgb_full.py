import sys
import struct
import json
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageOps

try:
    from PIL import ImageCms
    IMAGECMS_AVAILABLE = True
except Exception:
    IMAGECMS_AVAILABLE = False

# ---------- Utilities ----------
def align4(n: int) -> int:
    return (n + 3) & ~3

def pack_f32_array(floats):
    return struct.pack('<' + 'f' * len(floats), *floats)

def pack_u16_array(ints):
    return struct.pack('<' + 'H' * len(ints), *ints)

def build_plane_mesh(aspect_ratio=1.0):
    w = aspect_ratio
    h = 1.0
    positions = [
        -w/2, -h/2, 0.0,
         w/2, -h/2, 0.0,
         w/2,  h/2, 0.0,
        -w/2,  h/2, 0.0,
    ]
    normals = [0.0, 0.0, 1.0] * 4
    uvs = [
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    ]
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, uvs, indices

# ---------- Image processing ----------
def ensure_srgb_and_oriented(pil_image: Image.Image, srgb_icc_path: str = None) -> Image.Image:
    # Apply EXIF orientation
    img = ImageOps.exif_transpose(pil_image)

    # If ImageCms available and either an explicit srgb_icc_path is provided or the image has an embedded profile,
    # attempt profileToProfile conversion to sRGB.
    try:
        if IMAGECMS_AVAILABLE:
            src_profile = None
            dst_profile = None
            # if image has embedded icc_profile bytes, create a profile object
            icc_bytes = img.info.get("icc_profile")
            if icc_bytes:
                try:
                    src_profile = ImageCms.ImageCmsProfile(BytesIO(icc_bytes))
                except Exception:
                    src_profile = None
            # destination profile: either provided file or common sRGB profile from ImageCms
            if srgb_icc_path:
                try:
                    dst_profile = ImageCms.ImageCmsProfile(srgb_icc_path)
                except Exception:
                    dst_profile = None
            else:
                try:
                    dst_profile = ImageCms.createProfile("sRGB")
                except Exception:
                    dst_profile = None

            if dst_profile:
                if src_profile:
                    # convert from embedded profile -> sRGB
                    transform = ImageCms.buildTransformFromOpenProfiles(src_profile, dst_profile, img.mode, "RGB")
                    img = ImageCms.applyTransform(img, transform)
                    return img.convert("RGB")
                else:
                    # No embedded profile; convert to RGB (assume image pixels are already sRGB-ish)
                    return img.convert("RGB")
        # Fallback
        return img.convert("RGB")
    except Exception:
        # Any error, fallback to simple convert
        return img.convert("RGB")

# ---------- GLB builder ----------
def create_glb_bytes(image_bytes: bytes, image_mime: str, aspect_ratio: float = 1.0) -> bytes:
    positions, normals, texcoords, indices = build_plane_mesh(aspect_ratio)

    pos_b = pack_f32_array(positions)
    norm_b = pack_f32_array(normals)
    uv_b = pack_f32_array(texcoords)
    idx_b = pack_u16_array(indices)

    # Build binary blobs with 4-byte alignment
    chunks = []
    offset = 0
    def add_chunk(data: bytes):
        nonlocal offset
        start = offset
        chunks.append((start, data))
        offset += len(data)
        offset = align4(offset)

    add_chunk(idx_b)
    add_chunk(pos_b)
    add_chunk(norm_b)
    add_chunk(uv_b)
    add_chunk(image_bytes)

    total_bin_len = align4(offset)

    # Offsets
    indices_offset = chunks[0][0]
    positions_offset = chunks[1][0]
    normals_offset = chunks[2][0]
    texcoords_offset = chunks[3][0]
    image_offset = chunks[4][0]

    def bv_entry(byte_length, byte_offset, target=None):
        ent = {"buffer": 0, "byteOffset": byte_offset, "byteLength": byte_length}
        if target is not None:
            ent["target"] = target
        return ent

    bufferViews = [
        bv_entry(len(idx_b), indices_offset, target=34963),  # ELEMENT_ARRAY_BUFFER
        bv_entry(len(pos_b), positions_offset, target=34962),  # ARRAY_BUFFER
        bv_entry(len(norm_b), normals_offset, target=34962),
        bv_entry(len(uv_b), texcoords_offset, target=34962),
        bv_entry(len(image_bytes), image_offset, target=None),
    ]

    xs = positions[0::3]; ys = positions[1::3]; zs = positions[2::3]
    accessors = [
        {"bufferView": 0, "byteOffset": 0, "componentType": 5123, "count": len(indices), "type": "SCALAR", "max":[max(indices)], "min":[min(indices)]},
        {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3", "max":[max(xs), max(ys), max(zs)], "min":[min(xs), min(ys), min(zs)]},
        {"bufferView": 2, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
        {"bufferView": 3, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC2"},
    ]

    primitive = {"attributes": {"POSITION": 1, "NORMAL": 2, "TEXCOORD_0": 3}, "indices": 0, "mode": 4}
    mesh = {"primitives":[primitive], "name":"ImagePlane"}
    material = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}, "name":"ImageMaterial"}
    texture = {"source": 0, "name":"ImageTexture"}
    image_entry = {"bufferView": 4, "mimeType": image_mime, "name":"EmbeddedImage"}

    node = {"mesh": 0, "name":"PlaneNode"}
    scene = {"nodes":[0]}

    gltf = {
        "asset": {"generator":"python-jpg-to-glb-full", "version":"2.0"},
        "scenes":[scene],
        "scene":0,
        "nodes":[node],
        "meshes":[mesh],
        "materials":[material],
        "textures":[texture],
        "images":[image_entry],
        "accessors":accessors,
        "bufferViews":bufferViews,
        "buffers":[{"byteLength": total_bin_len}],
    }

    json_chunk = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode('utf-8')
    json_chunk_padded = json_chunk + b' ' * (align4(len(json_chunk)) - len(json_chunk))

    # Build binary blob
    bin_blob = bytearray(total_bin_len)
    for start, data in chunks:
        bin_blob[start:start+len(data)] = data

    # GLB header
    json_len = len(json_chunk_padded)
    bin_len = len(bin_blob)
    total_len = 12 + 8 + json_len + 8 + bin_len

    glb = bytearray()
    glb += struct.pack('<I', 0x46546C67)  # 'glTF'
    glb += struct.pack('<I', 2)           # version
    glb += struct.pack('<I', total_len)

    glb += struct.pack('<I', json_len)
    glb += struct.pack('<4s', b'JSON')
    glb += json_chunk_padded

    glb += struct.pack('<I', bin_len)
    glb += struct.pack('<4s', b'BIN\x00')
    glb += bin_blob

    return bytes(glb)

# ---------- Main ----------
def main():
    if len(sys.argv) < 3:
        print("Usage: python jpg_to_glb_srgb_full.py input_dir output_dir [optional_srgb_icc_path]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    srgb_icc = sys.argv[3] if len(sys.argv) > 3 else None
    output_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    if not jpgs:
        print("No JPG/JPEG files found in", input_dir)
        sys.exit(0)

    for jpg in jpgs:
        print("Processing", jpg.name)
        with Image.open(jpg) as im:
            proc = ensure_srgb_and_oriented(im, srgb_icc)
            w, h = proc.size
            aspect = (w / h) if h != 0 else 1.0
            # Re-encode to JPEG to embed final pixels
            buf = BytesIO()
            proc.save(buf, format="JPEG", quality=95, subsampling=0)
            image_bytes = buf.getvalue()

        mime = "image/jpeg"
        glb = create_glb_bytes(image_bytes, mime, aspect_ratio=aspect)
        out_path = output_dir / (jpg.stem + ".glb")
        out_path.write_bytes(glb)
        print("Wrote", out_path)

    print("All done.")

if __name__ == "__main__":
    main()
