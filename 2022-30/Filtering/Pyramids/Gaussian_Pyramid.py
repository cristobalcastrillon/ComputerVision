import sys
import cv2 as cv
import numpy as np

def downsample(src_img, kernel):
    '''
    Esta función hace downsampling de una imagen a una imagen de dimensiones w/2 * h/2 si la original es par; 
    floor(w/2) + 1 * floor(h/2) + 1 si es impar. 
    '''

    # Convolucionando la imagen original con un filtro cuyo kernel se recibe como parametro.
    src_img = cv.filter2D(src = src_img, ddepth = -1, kernel = kernel)

    src_height, src_width, _ = src_img.shape

    # Calculando las dimensiones espaciales de la matriz de destino.
    if(src_height % 2 == 0):
        dst_height = src_height / 2
    else:
        dst_height = (floor(src_height) / 2) + 1

    if(src_width % 2 == 0):
        dst_width = src_height / 2
    else:
        dst_width = (floor(src_height) / 2) + 1

    # Definiendo la matriz de destino y sus respectivos iteradores por dimensión.
    dst_img = np.zeros((int(dst_height), int(dst_width), 3), dtype = "uint8")
    dst_height_it = 0
    dst_width_it = 0

    # Iterar sobre la imagen descartando las filas y columnas de índice par.
    for i in range(src_height):
        if (i % 2 == 0):
            continue
        for j in range(src_width):
            if (j % 2 == 0):
                continue

            dst_img[dst_height_it % int(dst_height), dst_width_it % int(dst_width)] = src_img[i][j]

            dst_width_it += 1
        dst_height_it += 1
        
    return dst_img


def upsample(src_img, kernel):
    '''
    Esta función hace upsampling de una imagen a una imagen de dimensiones w/2 * h/2 si la original es par; 
    floor(w/2) + 1 * floor(h/2) + 1 si es impar. 
    '''

    src_height, src_width, _ = src_img.shape

    # Calculando las dimensiones espaciales de la matriz de destino.
    dst_height = src_height * 2
    dst_width = src_width * 2

    # Definiendo la matriz de destino.
    dst_img = np.zeros((dst_height, dst_width, 3), dtype = "uint8")
    
    src_height_it = 0
    src_width_it = 0
    
    for i in range(dst_height):
        if (i % 2 == 0):
            continue        
        for j in range(dst_width):
            if (j % 2 == 0):
                continue
            dst_img[i, j] = src_img[src_width_it % src_width][src_height_it % src_height]
            
            src_height_it += 1
        src_width_it += 1
        
    # Convolucionando la imagen original con un filtro cuyo kernel se recibe como parametro.
    dst_img = cv.filter2D(src = dst_img, ddepth = -1, kernel = kernel)    

    return dst_img

# -----------------------------------------------------------------------------------------------
# Pirámide Gaussiana
src = cv.imread('chicky_512.png')

if src is None:
    print ('Error opening image!')

kernel1 = np.array([[1/256, 4/256, 6/256, 4/256, 1/256],
                   [4/256, 16/256, 24/256, 16/256, 4/256],
                   [6/256, 24/256, 36/256, 24/256, 6/256],
                   [4/256, 16/256, 24/256, 16/256, 4/256],
                   [1/256, 4/256, 6/256, 4/256, 1/256]])

kernel2 = np.array([[4/256, 16/256, 24/256, 16/256, 4/256],
                   [16/256, 64/256, 96/256, 64/256, 16/256],
                   [24/256, 96/256, 144/256, 96/256, 24/256],
                   [16/256, 64/256, 96/256, 64/256, 16/256],
                   [4/256, 16/256, 24/256, 16/256, 4/256]])

# Nivel +1
down_1 = downsample(src, kernel1)
cv.imwrite('chicky_256.png', down_1)

# Nivel +2
down_2 = downsample(down_1, kernel1)
cv.imwrite('chicky_128.png', down_2)

# Nivel +3
down_3 = downsample(down_2, kernel1)
cv.imwrite('chicky_64.png', down_3)

# Nivel -1
up_1 = upsample(src, kernel2)
cv.imwrite('chicky_1024.png', up_1)

# Nivel -2
up_2 = upsample(up_1, kernel2)
cv.imwrite('chicky_2048.png', up_2)

# Nivel -3
up_3 = upsample(up_2, kernel2)
cv.imwrite('chicky_4096.png', up_3)

# Pirámide Laplaciana

# Laplace 64
cv.imwrite('chicky_laplace_64.png', down_3)

# Nivel +3 a Nivel +2
up_laplace_1 = upsample(down_3, kernel2)
up_laplace_1 = cv.subtract(down_2, up_laplace_1)
cv.imwrite('chicky_laplace_128.png', up_laplace_1)

# Nivel +2 a Nivel +1
up_laplace_2 = upsample(down_2, kernel2)
up_laplace_2 = cv.subtract(down_1, up_laplace_2)
cv.imwrite('chicky_laplace_256.png', up_laplace_2)

# Nivel +1 a Nivel 0
up_laplace_3 = upsample(down_1, kernel2)
up_laplace_3 = cv.subtract(src, up_laplace_3)
cv.imwrite('chicky_laplace_512.png', up_laplace_3)

# Nivel 0 a Nivel +1
up_laplace_4 = upsample(src, kernel2)
up_laplace_4 = cv.subtract(up_1, up_laplace_4)
cv.imwrite('chicky_laplace_1024.png', up_laplace_4)

# Nivel +1 a Nivel +2
up_laplace_5 = upsample(up_1, kernel2)
up_laplace_5 = cv.subtract(up_2, up_laplace_5)
cv.imwrite('chicky_laplace_2048.png', up_laplace_5)

# Nivel +2 a Nivel +3
up_laplace_6 = upsample(up_2, kernel2)
up_laplace_6 = cv.subtract(up_3, up_laplace_6)
cv.imwrite('chicky_laplace_4096.png', up_laplace_6)