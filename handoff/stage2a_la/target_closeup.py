"""Crop the complete visible target without adding pixels or changing its geometry."""
import math


def crop_bounds(mask, gsd):
    width, height = mask.size
    bounds = mask.getbbox()
    assert bounds is not None and width == height and gsd > 0
    left, top, right, bottom = bounds
    # shortcut: edge-touching masks retain their source crop; wider source recovery is a separate experiment.
    if left == 0 or top == 0 or right == width or bottom == height:
        return (0, 0, width, height)
    side = min(width, max(math.ceil(40 / gsd), math.ceil(1.4 * max(right-left, bottom-top))))
    x = max(0, min(width-side, (left+right-side)//2))
    y = max(0, min(height-side, (top+bottom-side)//2))
    assert x <= left and y <= top and x+side >= right and y+side >= bottom
    return (x, y, x+side, y+side)


def self_check():
    from PIL import Image, ImageDraw
    for size, box, gsd in [(256, (85, 104, 150, 132), .305),
                           (256, (5, 10, 33, 170), .467),
                           (256, (0, 25, 30, 60), .305),
                           (256, (10, 10, 245, 230), .305)]:
        mask = Image.new('L', (size, size)); ImageDraw.Draw(mask).rectangle(box, fill=255)
        bounds = crop_bounds(mask, gsd)
        assert sum(mask.crop(bounds).getdata()) == sum(mask.getdata())
        assert bounds[2]-bounds[0] == bounds[3]-bounds[1]
        if box[0] == 0: assert bounds == (0, 0, size, size)
    assert crop_bounds(mask, gsd) == (0, 0, size, size)
    print('PASS complete visible mask, edge fallback, square crop, bounded context')


if __name__ == '__main__': self_check()
