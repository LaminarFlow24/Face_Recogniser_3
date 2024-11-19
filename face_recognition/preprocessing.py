from PIL import Image

exif_orientation_tag = 0x0112
exif_transpose_sequences = [
    [],
    [],
    [Image.FLIP_LEFT_RIGHT],
    [Image.ROTATE_180],
    [Image.FLIP_TOP_BOTTOM],
    [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],
    [Image.ROTATE_270],
    [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],
    [Image.ROTATE_90],
]


class ExifOrientationNormalize(object):
    def __call__(self, img):
        if 'parsed_exif' in img.info and exif_orientation_tag in img.info['parsed_exif']:
            orientation = img.info['parsed_exif'][exif_orientation_tag]
            for trans in exif_transpose_sequences[orientation]:
                img = img.transpose(trans)
        return img
