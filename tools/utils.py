import numpy as np
from functools import reduce

from PIL import Image
import cv2
from scipy.spatial import distance
from skimage.color import rgb2lab, lab2rgb

# find connection in the specified sequence, center 29 is in the position 15
limbSeq=[[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],
         [10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],
         [1,16],[16,18],[3,17],[6,18]]

# the middle joints heatmap correpondence
hmapIdx=[[31,32],[39,40],[33,34],[35,36],[41,42],[43,44],[19,20],[21,22],
         [23,24],[25,26],[27,28],[29,30],[47,48],[49,50],[53,54],[51,52],
         [55,56],[37,38],[45,46]]

# visualize
colors=[[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],
        [0,255,0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
        [85, 0, 255],
        [170,0,255],[255, 0, 255], [255, 0, 170], [255, 0, 85]]


def pad_right_down_corner(img,stride,pad_value):   # 8   128
    h=img.shape[0]
    w=img.shape[1]

    pad=4*[None]
    pad[0]=0  # up
    pad[1]=0  # left
    pad[2]=0 if (h%stride==0) else stride-(h%stride)  # down
    pad[3]=0 if (w%stride==0) else stride-(w%stride)  # right

    img_padded=img
    pad_up=np.tile(img_padded[0:1,:,:]*0+pad_value,(pad[0],1,1))
    img_padded=np.concatenate((pad_up,img_padded),axis=0)
    pad_left=np.tile(img_padded[:,0:1,:]*0+pad_value,(1,pad[1],1))
    img_padded=np.concatenate((pad_left,img_padded),axis=1)
    pad_down=np.tile(img_padded[-2:-1,:,:]*0+pad_value,(pad[2],1,1))
    img_padded=np.concatenate((img_padded,pad_down),axis=0)
    pad_right=np.tile(img_padded[:,-2:-1,:]*0+pad_value,(1,pad[3],1))
    img_padded=np.concatenate((img_padded,pad_right),axis=1)

    return img_padded,pad

def compose(*funcs):
    #Compose arbitrarily many functions, evaluated left to right.
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f,g: lambda *a,**kw: g(f(*a,**kw)),funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image,size):
    #Resize image with unchanged aspect ratio using padding.
    image_w,image_h=image.size
    w,h=size
    new_w=int(image_w*min(w*1.0/image_w,h*1.0/image_h))
    new_h=int(image_h*min(w*1.0/image_w,h*1.0/image_h))
    resized_image=image.resize((new_w,new_h),Image.BICUBIC)
    #Padding the image
    boxed_image=Image.new('RGB',size,(128,128,128))     #New image with a size & color(128,128,128)
    boxed_image.paste(resized_image,((w-new_w)//2,(h-new_h)//2))       #Pasting resized_image over newly created image at given coordinates.
    return boxed_image

def compare_color_histograms(image1, image2):
    # Convert images to the LAB color space
    image1_lab = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    image2_lab = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    # Calculate the histograms for the LAB channels
    hist1 = cv2.calcHist([image1_lab], [1, 2], None, [128, 128], [0, 256, 0, 256])
    hist2 = cv2.calcHist([image2_lab], [1, 2], None, [128, 128], [0, 256, 0, 256])

    # Normalize the histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Calculate the Bhattacharyya similarity
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return similarity

def crop_minAreaRect(img, rect):

    # # rotate img
    # rows,cols = img.shape[0], img.shape[1]
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    # img_rot = cv2.warpAffine(img,M,(cols,rows))

    # # rotate bounding box
    # rect0 = (rect[0], rect[1], 0.0) 
    # box = cv2.boxPoints(rect0)
    # pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    # pts[pts < 0] = 0

    # # crop
    # img_crop = img_rot[pts[1][1]:pts[0][1], 
    #                    pts[1][0]:pts[2][0]]

    # return img_crop

    width = np.linalg.norm(np.array(rect[0]) - np.array(rect[1]))
    height = np.linalg.norm(np.array(rect[1]) - np.array(rect[2]))

    # Calculate the rotation angle to make the rectangle vertical
    angle = 0
    if width > height:
        angle = 90  # Rotate 90 degrees to make it vertical

    # Convert the points to a NumPy array
    rect_points = np.array(rect, dtype=np.float32)

    # Get the width and height of the rotated rectangle
    if angle == 90:
        rotated_width, rotated_height = height, width
    else:
        rotated_width, rotated_height = width, height

    # Calculate the destination points for the perspective transformation
    dst_points = np.array([[0, 0], [rotated_width, 0], [rotated_width, rotated_height], [0, rotated_height]], dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(rect_points, dst_points)

    # Warp the image using the perspective transformation matrix
    warped_image = cv2.warpPerspective(img, matrix, (int(rotated_width), int(rotated_height)))

    return warped_image

def angle_between_vectors(v1, v2):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes (lengths) of the vectors
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle between the vectors using the dot product and magnitudes
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians using the arccosine function (inverse cosine)
    angle_in_radians = np.arccos(cosine_angle)

    # Convert the angle to degrees if desired
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees, angle_in_radians


iscc_nbs_color_dict = {
    "#ffb5ba": "Vivid_Pink",
    "#ea9399": "Strong_Pink",
    "#e4717a": "Deep_Pink",
    "#f9ccca": "Light_Pink",
    "#dea5a4": "Moderate_Pink",
    "#c08081": "Dark_Pink",
    "#ead8d7": "Pale_Pink",
    "#c4aead": "Grayish_Pink",
    "#eae3e1": "Pinkish_White",
    "#c1b6b3": "Pinkish_Gray",
    "#be0032": "Vivid_Red",
    "#bc3f4a": "Strong_Red",
    "#841b2d": "Deep_Red",
    "#5c0923": "Very_Deep_Red",
    "#ab4e52": "Moderate_Red",
    "#722f37": "Dark_Red",
    "#3f1728": "Very_Dark_Red",
    "#ad8884": "Light_Grayish_Red",
    "#905d5d": "Grayish_Red",
    "#543d3f": "Dark_Grayish_Red",
    "#2e1d21": "Blackish_Red",
    "#8f817f": "Reddish_Gray",
    "#5c504f": "Dark_Reddish_Gray",
    "#282022": "Reddish_Black",
    "#ffb7a5": "Vivid_Yellowish_Pink",
    "#f99379": "Strong_Yellowish_Pink",
    "#e66721": "Deep_Yellowish_Pink",
    "#f4c2c2": "Light_Yellowish_Pink",
    "#d9a6a9": "Moderate_Yellowish_Pink",
    "#c48379": "Dark_Yellowish_Pink",
    "#ecd5c5": "Pale_Yellowish_Pink",
    "#c7ada3": "Grayish_Yellowish_Pink",
    "#c2ac99": "Brownish_Pink",
    "#e25822": "Vivid_Reddish_Orange",
    "#d9603b": "Strong_Reddish_Orange",
    "#aa381e": "Deep_Reddish_Orange",
    "#cb6d51": "Moderate_Reddish_Orange",
    "#9e4732": "Dark_Reddish_Orange",
    "#b4745e": "Grayish_Reddish_Orange",
    "#882d17": "Strong_Reddish_Brown",
    "#56070c": "Deep_Reddish_Brown",
    "#a87c6d": "Light_Reddish_Brown",
    "#79443b": "Moderate_Reddish_Brown",
    "#3e1d1e": "Dark_Reddish_Brown",
    "#977f73": "Light_Grayish_Reddish_Brown",
    "#674c47": "Grayish_Reddish_Brown",
    "#43302e": "Dark_Grayish_Reddish_Brown",
    "#f38400": "Vivid_Orange",
    "#fd943f": "Brilliant_Orange",
    "#ed872d": "Strong_Orange",
    "#be6516": "Deep_Orange",
    "#fab57f": "Light_Orange",
    "#d99058": "Moderate_Orange",
    "#ae6938": "Brownish_Orange",
    "#80461b": "Strong_Brown",
    "#593319": "Deep_Brown",
    "#a67b5b": "Light_Brown",
    "#6f4e37": "Moderate_Brown",
    "#422518": "Dark_Brown",
    "#958070": "Light_Grayish_Brown",
    "#635147": "Grayish_Brown",
    "#3e322c": "Dark_Grayish_Brown",
    "#8e8279": "Light_Brownish_Gray",
    "#5b504f": "Brownish_Gray",
    "#28201c": "Brownish_Black",
    "#f6a600": "Vivid_Orange_Yellow",
    "#ffc14f": "Brilliant_Orange_Yellow",
    "#eaa221": "Strong_Orange_Yellow",
    "#c98500": "Deep_Orange_Yellow",
    "#fbc97f": "Light_Orange_Yellow",
    "#e3a857": "Moderate_Orange_Yellow",
    "#be8a3d": "Dark_Orange_Yellow",
    "#fad6a5": "Pale_Orange_Yellow",
    "#996515": "Strong_Yellowish_Brown",
    "#654522": "Deep_Yellowish_Brown",
    "#c19a6b": "Light_Yellowish_Brown",
    "#826644": "Moderate_Yellowish_Brown",
    "#4b3621": "Dark_Yellowish_Brown",
    "#ae9b82": "Light_Grayish_Yellowish_Brown",
    "#7e6d5a": "Grayish_Yellowish_Brown",
    "#483c32": "Dark_Grayish_Yellowish_Brown",
    "#f3c300": "Vivid_Yellow",
    "#fada5e": "Brilliant_Yellow",
    "#d4af37": "Strong_Yellow",
    "#af8d13": "Deep_Yellow",
    "#f8de7e": "Light_Yellow",
    "#c9ae5d": "Moderate_Yellow",
    "#ab9144": "Dark_Yellow",
    "#f3e5ab": "Pale_Yellow",
    "#c2b280": "Grayish_Yellow",
    "#a18f60": "Dark_Grayish_Yellow",
    "#f0ead6": "Yellowish_White",
    "#bfb8a5": "Yellowish_Gray",
    "#967117": "Light_Olive_Brown",
    "#6c541e": "Moderate_Olive_Brown",
    "#3b3121": "Dark_Olive_Brown",
    "#dcd300": "Vivid_Greenish_Yellow",
    "#e9e450": "Brilliant_Greenish_Yellow",
    "#beb72e": "Strong_Greenish_Yellow",
    "#9b9400": "Deep_Greenish_Yellow",
    "#eae679": "Light_Greenish_Yellow",
    "#b9b459": "Moderate_Greenish_Yellow",
    "#98943e": "Dark_Greenish_Yellow",
    "#ebe8a4": "Pale_Greenish_Yellow",
    "#b9b57d": "Grayish_Greenish_Yellow",
    "#867e36": "Light_Olive",
    "#665d1e": "Moderate_Olive",
    "#403d21": "Dark_Olive",
    "#8c8767": "Light_Grayish_Olive",
    "#5b5842": "Grayish_Olive",
    "#363527": "Dark_Grayish_Olive",
    "#8a8776": "Light_Olive_Gray",
    "#57554c": "Olive_Gray",
    "#25241d": "Olive_Black",
    "#8db600": "Vivid_Yellow_Green",
    "#bdda57": "Brilliant_Yellow_Green",
    "#7e9f2e": "Strong_Yellow_Green",
    "#467129": "Deep_Yellow_Green",
    "#c9dc89": "Light_Yellow_Green",
    "#8a9a5b": "Moderate_Yellow_Green",
    "#dadfb7": "Pale_Yellow_Green",
    "#8f9779": "Grayish_Yellow_Green",
    "#404f00": "Strong_Olive_Green",
    "#232f00": "Deep_Olive_Green",
    "#4a5d23": "Moderate_Olive_Green",
    "#2b3d26": "Dark_Olive_Green",
    "#515744": "Grayish_Olive_Green",
    "#31362b": "Dark_Grayish_Olive_Green",
    "#27a64c": "Vivid_Yellowish_Green",
    "#83d37d": "Brilliant_Yellowish_Green",
    "#44944a": "Strong_Yellowish_Green",
    "#00622d": "Deep_Yellowish_Green",
    "#003118": "Very_Deep_Yellowish_Green",
    "#b6e5af": "Very_Light_Yellowish_Green",
    "#93c592": "Light_Yellowish_Green",
    "#679267": "Moderate_Yellowish_Green",
    "#355e3b": "Dark_Yellowish_Green",
    "#173620": "Very_Dark_Yellowish_Green",
    "#008856": "Vivid_Green",
    "#3eb489": "Brilliant_Green",
    "#007959": "Strong_Green",
    "#00543d": "Deep_Green",
    "#8ed1b2": "Very_Light_Green",
    "#6aab8e": "Light_Green",
    "#3b7861": "Moderate_Green",
    "#1b4d3e": "Dark_Green",
    "#1c352d": "Very_Dark_Green",
    "#c7e6d7": "Very_Pale_Green",
    "#8da399": "Pale_Green",
    "#5e716a": "Grayish_Green",
    "#3a4b47": "Dark_Grayish_Green",
    "#1a2421": "Blackish_Green",
    "#dfede8": "Greenish_White",
    "#b2beb5": "Light_Greenish_Gray",
    "#7d8984": "Greenish_Gray",
    "#4e5755": "Dark_Greenish_Gray",
    "#1e2321": "Greenish_Black",
    "#008882": "Vivid_Bluish_Green",
    "#00a693": "Brilliant_Bluish_Green",
    "#007a74": "Strong_Bluish_Green",
    "#00443f": "Deep_Bluish_Green",
    "#96ded1": "Very_Light_Bluish_Green",
    "#66ada4": "Light_Bluish_Green",
    "#317873": "Moderate_Bluish_Green",
    "#004b49": "Dark_Bluish_Green",
    "#002a29": "Very_Dark_Bluish_Green",
    "#0085a1": "Vivid_Greenish_Blue",
    "#239eba": "Brilliant_Greenish_Blue",
    "#007791": "Strong_Greenish_Blue",
    "#2e8495": "Deep_Greenish_Blue",
    "#9cd1dc": "Very_Light_Greenish_Blue",
    "#66aabc": "Light_Greenish_Blue",
    "#367588": "Moderate_Greenish_Blue",
    "#004958": "Dark_Greenish_Blue",
    "#002e3b": "Very_Dark_Greenish_Blue",
    "#00a1c2": "Vivid_Blue",
    "#4997d0": "Brilliant_Blue",
    "#0067a5": "Strong_Blue",
    "#00416a": "Deep_Blue",
    "#a1caf1": "Very_Light_Blue",
    "#70a3cc": "Light_Blue",
    "#436b95": "Moderate_Blue",
    "#00304e": "Dark_Blue",
    "#bcd4e6": "Very_Pale_Blue",
    "#91a3b0": "Pale_Blue",
    "#536878": "Grayish_Blue",
    "#36454f": "Dark_Grayish_Blue",
    "#202830": "Blackish_Blue",
    "#e9e9ed": "Bluish_White",
    "#b4bcc0": "Light_Bluish_Gray",
    "#81878b": "Bluish_Gray",
    "#51585e": "Dark_Bluish_Gray",
    "#202428": "Bluish_Black",
    "#30267a": "Vivid_Purplish_Blue",
    "#6c79b8": "Brilliant_Purplish_Blue",
    "#545aa7": "Strong_Purplish_Blue",
    "#272458": "Deep_Purplish_Blue",
    "#b3bce2": "Very_Light_Purplish_Blue",
    "#8791bf": "Light_Purplish_Blue",
    "#4e5180": "Moderate_Purplish_Blue",
    "#252440": "Dark_Purplish_Blue",
    "#c0c8e1": "Very_Pale_Purplish_Blue",
    "#8c92ac": "Pale_Purplish_Blue",
    "#4c516d": "Grayish_Purplish_Blue",
    "#9065ca": "Vivid_Violet",
    "#7e73b8": "Brilliant_Violet",
    "#604e97": "Strong_Violet",
    "#32174d": "Deep_Violet",
    "#dcd0ff": "Very_Light_Violet",
    "#8c82b6": "Light_Violet",
    "#604e81": "Moderate_Violet",
    "#2f2140": "Dark_Violet",
    "#c4c3dd": "Very_Pale_Violet",
    "#9690ab": "Pale_Violet",
    "#554c69": "Grayish_Violet",
    "#9a4eae": "Vivid_Purple",
    "#d399e6": "Brilliant_Purple",
    "#875692": "Strong_Purple",
    "#602f6b": "Deep_Purple",
    "#401a4c": "Very_Deep_Purple",
    "#d5badb": "Very_Light_Purple",
    "#b695c0": "Light_Purple",
    "#86608e": "Moderate_Purple",
    "#563c5c": "Dark_Purple",
    "#301934": "Very_Dark_Purple",
    "#d6cadd": "Very_Pale_Purple",
    "#aa98a9": "Pale_Purple",
    "#796878": "Grayish_Purple",
    "#50404d": "Dark_Grayish_Purple",
    "#291e29": "Blackish_Purple",
    "#e8e3e5": "Purplish_White",
    "#bfb9bd": "Light_Purplish_Gray",
    "#8b8589": "Purplish_Gray",
    "#5d555b": "Dark_Purplish_Gray",
    "#242124": "Purplish_Black",
    "#870074": "Vivid_Reddish_Purple",
    "#9e4f88": "Strong_Reddish_Purple",
    "#702963": "Deep_Reddish_Purple",
    "#54194e": "Very_Deep_Reddish_Purple",
    "#b784a7": "Light_Reddish_Purple",
    "#915c83": "Moderate_Reddish_Purple",
    "#5d3954": "Dark_Reddish_Purple",
    "#341731": "Very_Dark_Reddish_Purple",
    "#aa8a9e": "Pale_Reddish_Purple",
    "#836479": "Grayish_Reddish_Purple",
    "#ffc8d6": "Brilliant_Purplish_Pink",
    "#e68fac": "Strong_Purplish_Pink",
    "#de6fa1": "Deep_Purplish_Pink",
    "#efbbcc": "Light_Purplish_Pink",
    "#d597ae": "Moderate_Purplish_Pink",
    "#c17e91": "Dark_Purplish_Pink",
    "#e8ccd7": "Pale_Purplish_Pink",
    "#c3a6b1": "Grayish_Purplish_Pink",
    "#ce4676": "Vivid_Purplish_Red",
    "#b3446c": "Strong_Purplish_Red",
    "#78184a": "Deep_Purplish_Red",
    "#54133b": "Very_Deep_Purplish_Red",
    "#a8516e": "Moderate_Purplish_Red",
    "#673147": "Dark_Purplish_Red",
    "#38152c": "Very_Dark_Purplish_Red",
    "#af868e": "Light_Grayish_Purplish_Red",
    "#915f6d": "Grayish_Purplish_Red",
    "#f2f3f4": "White",
    "#b9b8b5": "Light_Gray",
    "#848482": "Medium_Gray",
    "#555555": "Dark_Gray",
    "#222222": "Black"
}

def modify_keys(dictionary, key_func):
    modified_dict = {}
    for old_key, value in dictionary.items():
        new_key = key_func(old_key)
        modified_dict[new_key] = value
    return modified_dict


def color_code_to_rgb(color_code):
    # Remove the '#' character if it exists
    color_code = color_code.lstrip('#')

    # Split the color code into its two 8-bit components (red and green)
    red = int(color_code[0:2], 16)
    green = int(color_code[2:4], 16)

    # Calculate the blue component by taking the remaining 8 bits
    blue = int(color_code[4:], 16)

    # Return the RGB values as a tuple
    return (red, green, blue)

iscc_nbs_color_dict = modify_keys(iscc_nbs_color_dict, color_code_to_rgb)

lut1 = [rgb2lab(np.array(rgb, dtype="float32") / 255) for rgb in iscc_nbs_color_dict.keys()]

# Step 2: Create LUT2 with dominant hues
# Assuming iscc_nbs_color_names is a list of color names for ISCC-NBS colors
lut2 = {tuple(rgb): color_name for rgb, color_name in iscc_nbs_color_dict.items()}

# Step 3: Find the closest ISCC-NBS color for a given pixel
def find_closest_color(pixel_lab):
    distances = [distance.euclidean(pixel_lab, lab) for lab in lut1]
    closest_index = np.argmin(distances)
    return lut2[tuple(list(iscc_nbs_color_dict.keys())[closest_index])]