import numpy as np
import os
def extract(imag):
    with open(imag , 'rb') as f:   #ouvrir le fichier en mode lecture binaire
        magic_num = f.readline().strip()     # strip() pour supprimer les espaces 
        width , height = map(int , f.readline().split())
        max_val = int(f.readline())
        pixls =[]
        for _ in range(width):
            row = []
            for _ in range(height):
                byte = f.read(1)
                pix_val = int.from_bytes(byte , byteorder='big')
                row.append(pix_val)
            pixls.append(row)
    return np.array(pixls)


def Lecture_fichiers(chemin_acces):
    images = []
    target = []
    for folder_name in os.listdir(chemin_acces):
        folder_path = os.path.join(chemin_acces , folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path , image_name)
                if image_path.endswith('.pgm'):
                    image = extract(image_path)
                    images.append(image)
                    target.append(int(folder_name[1:]))
    return images , target 

def LBP_function (image):
    longueur , largeur = image.shape[0] , image.shape[1]
    lbp_image = np.zeros((longueur - 2 , largeur - 2),int)
    for i in range(1 , longueur -1):
        for j in range(1 , largeur -1):
            pixels_voisins = [image[i-1][j-1] , image[i-1][j] , image[i-1][j+1] , image[i][j+1],
                                  image[i+1][j+1] , image[i+1][j] , image[i+1][j-1] , image[i][j-1]]
            for n in range(len(pixels_voisins)):
                if pixels_voisins[n] >= image[i][j]:
                    lbp_image[i-1][j-1] = lbp_image[i-1][j-1]+2**n 
    return lbp_image 


def createPGM(matrix):
    with open('image.pgm', 'wb') as f:
        f.write(b'P5\n')
        f.write('{} {}\n'.format(len(matrix),len(matrix[0])).encode())
        f.write(b'255\n')
        f.write(matrix.astype(np.uint8).tobytes())


def LBP_function_final(liste_imag):
    list_lbp_img = []
    for imag in liste_imag:
        lbp = LBP_function(imag)
        list_lbp_img.append(lbp)
    return list_lbp_img

def histogramme(img_lbp , window = 10):
    hist_courant = []
    longueur , largeur = img_lbp.shape[0] , img_lbp.shape[1]
    for i in range(0 , longueur-window+1 , window):
        for j in range(0 , largeur-window+1 , window):
            hist = [0]*256
            for x in range(i , i+window):
                for y in range(j , j+window):
                    val = img_lbp[x][y]
                    hist[val] += 1
            hist /=np.sum(hist)
            hist_courant.append(hist)
    total_hist = list(hist_courant[0])
    for h in range(1 , len(hist_courant)):
        total_hist += list( hist_courant[h])
    return total_hist


def histigramme_final(liste_img_lbp ,  window_ = 10):
    liste_hist = []
    for img_lbp in liste_img_lbp:
        liste_hist.append(histogramme(img_lbp , window_ ))
    return liste_hist


