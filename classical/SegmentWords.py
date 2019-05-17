import cv2
import itertools
import numpy as np

def get_start_(image):
    black_pixels = np.argwhere(image == 0)
    max_x = max(black_pixels[:,0])
    pixels_with_max_x = black_pixels[np.where(black_pixels[:,0] == max_x), :][0]
    min_y = min(pixels_with_max_x[:,1])
    return np.ravel(pixels_with_max_x[np.where(pixels_with_max_x[:,1] == min_y), :][0])

def get_start(image):
    black_pixels = np.argwhere(image == 0)
    min_y = min(black_pixels[:,1])
    pixels_with_min_y = black_pixels[np.where(black_pixels[:,1] == min_y), :][0]
    max_x = max(pixels_with_min_y[:,0])
    return np.ravel(pixels_with_min_y[np.where(pixels_with_min_y[:,0] == max_x), :][0])


def get_segments(image, kernel_size = 15):
    img = image.copy()[:,:,0]
    n, m = img.shape
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    img = cv2.bitwise_not(thresh)     
    _, markers = cv2.connectedComponents(img)
    component_count = np.amax(markers)
    segments = []    
    for comp in range(1, component_count + 1):
        mask = markers == comp
        component = img * mask
        if np.sum(component) / 255 < 100:
            continue
        component = cv2.bitwise_not(component)
        start = get_start(component)     
        start_points = [start]
        used_cells = []
        segments_points = []
        component_3 = np.stack((component,)*3, axis=-1)
        while len(start_points) > 0:
            start = start_points.pop(0)
            if (0 < start[0] - kernel_size < n):
                top_left = (start[1], start[0] - kernel_size)
            elif start[0] - kernel_size >= n:
                top_left = (start[1], n)
            elif start[0] - kernel_size <= 0:
                top_left = (start[1], 0)
                
            if (0 < start[1] + kernel_size < m):
                bottom_right = (start[1] + kernel_size, start[0])
            elif start[1] + kernel_size >= m:
                bottom_right = (m, start[0])
            elif start[1] + kernel_size <= 0:
                bottom_right = (0, start[0])

                
            if top_left is None or bottom_right is None:
                continue

            segments_points.append([top_left, bottom_right])
            top = max(0, start[0] - kernel_size)
            bottom = min(n, start[0])
            left = max(0, start[1])
            right = min(m, start[1] + kernel_size)

            segment = component[top:bottom, left:right]
            s_n, s_m = segment.shape
            if s_n < kernel_size:
                segment = cv2.copyMakeBorder(segment,kernel_size - s_n,0,0,0,cv2.BORDER_CONSTANT,value=255)
            if s_m < kernel_size:
                segment = cv2.copyMakeBorder(segment,0,0,0,kernel_size - s_m,cv2.BORDER_CONSTANT,value=255)

            segments_cells = list(itertools.product(range(top, bottom + 1), range(left, right + 1)))
            if not set(segments_cells).issubset(set(used_cells)): 
                used_cells += segments_cells
                segments.append(segment)
            else: 
                continue

            offset = 0
            if (0 <= start[0] - kernel_size - 1 <= n) and (0 <= start[1] + kernel_size <= m):
                top_line = component[top - 1,  left:right]   
                if 0 in top_line:
                    new_start = (start[0] - kernel_size - 1, start[1] + offset)
                    if new_start not in used_cells and new_start not in start_points:
                        start_points.append(new_start)

            if (0 <= start[0] +  kernel_size + 1 <= n) and (0 <= start[1] + kernel_size <= m):
                if bottom + 1 != n:
                    bot_line = component[bottom + 1,  left:right]
                    if 0 in bot_line:
                        new_start = (start[0] + kernel_size + 1, start[1] + offset)
                        if new_start not in used_cells and new_start not in start_points:
                            start_points.append(new_start)

            if (0 <= start[0] - kernel_size <= n) and (0 <= start[1] - kernel_size - 1 <= m):
                left_line = component[top: bottom,  left - 1] 
                if 0 in left_line:
                    new_start = (start[0] - offset, start[1] - kernel_size - 1)
                    if new_start not in used_cells and new_start not in start_points:
                        start_points.append(new_start)

            if (0 <= start[0] - kernel_size <= n) and (0 <= start[1] + kernel_size + 1 <= m):
                if right + 1 != m:   
                    right_line = component[top: bottom,  right + 1]
                    if 0 in right_line:
                        new_start = (start[0] - offset, start[1] + kernel_size + 1)
                        if new_start not in used_cells and new_start not in start_points:
                            start_points.append(new_start)

    return segments
    
    
def save_segments(path, segments, count = 0):
    for segment in segments:
        if np.sum(segment == 0) > 10:
            cv2.imwrite(os.path.join(path , 'segment_' + str(count) + '.png'), segment)
            count += 1
           

def get_starts(vect):
    inds = []
    for i, val in enumerate(vect):
        if val == 0 and i == 0:
            inds.append(i)
        if i > 0 and vect[i - 1] == 255 and val == 0:
            inds.append(i)
    return inds
