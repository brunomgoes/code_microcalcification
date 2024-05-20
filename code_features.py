import os
import time
import pydicom

import pandas as pd
import numpy as np
import mahotas as mt

from skimage import io, measure
from tkinter import filedialog
from scipy import stats

def get_roi(img_array, gt_array, label_array, file_path):
    img = np.copy(img_array)
    file_name = file_path
    gt = np.copy(gt_array)
    r_label = label_array
    
    roi_data = []
    roi_size = 100
    count = 0
    nrolimite= 0.7

    coluna = img.shape[0]
    linha = img.shape[1]
    extracoluna = coluna % roi_size
    extralinha = linha % roi_size
    coluna = coluna - extracoluna
    linha = linha - extralinha
    nrocolunas = int(np.round(coluna / roi_size))  # arrendonda para o numero mais baixo por exemplo 10/3 =3
    nrolinhas = int(np.round(linha / roi_size))
    area = roi_size*roi_size

    if file_name.find('_L_') > 0:
        #left
        for x in range(0, nrocolunas, 1):
            for y in range(0, nrolinhas, 1):
                a = x * roi_size
                b = (x * roi_size) + roi_size
                c = y * roi_size
                d = (y * roi_size) + roi_size
                aux = b - a
                aux2 = d - c

                list_item = {
                    'label': count,
                    'x': a,
                    'y': c,
                    'array': img[a:b, c:d],
                    'ground_truth': gt[a:b, c:d],
                    'r_label': r_label[a:b, c:d]
                }
                
                if (np.percentile(list_item['array'], 75) > 0): # seleciona os rois pelo percentil
                    count += 1
                    roi_data.append(list_item)

    else:
        #right
        for x in range(nrocolunas,0,-1):
            for y in range(nrolinhas, 0, -1):
                a = (x * roi_size)+extracoluna
                b = ((x * roi_size)+extracoluna) - roi_size
                c = (y * roi_size) + extralinha
                d = ((y * roi_size)+extralinha) - roi_size
                aux = a-b
                aux2 = c-d

                list_item = {
                    'label': count,
                    'x': b,
                    'y': d,
                    'array': img[b:a, d:c],
                    'ground_truth': gt[b:a, d:c],
                    'r_label': r_label[b:a, d:c]
                }
                
                # if (np.percentile(list_item['array'], 75) > 0): # seleciona os rois pelo percentil
                count +=1
                roi_data.append(list_item)

    return roi_data

def get_features(object_array, img_array, roi_label, roi_result):
    r_label, r_num = measure.label(object_array, return_num=True)
    r_props = measure.regionprops(r_label, intensity_image=img_array)

    micro_list = []
    ################# características dos objetos
    for item in r_props:
        micro_dict = {
            'roi_index': roi_label,
            'area': item.area,
            'a_bbox': item.area_bbox,
            'major_length': item.axis_major_length,
            'minor_length': item.axis_minor_length,
            'eccentricity': item.eccentricity,
            'equivalent_diameter_area': item.equivalent_diameter_area,
            'euler_number': item.euler_number,
            'extent': item.extent,
            'feret_diameter_max': item.feret_diameter_max,
            'i_max': item.intensity_max,
            'i_mean': item.intensity_mean,
            'i_min': item.intensity_min,
            'orientation': item.orientation,
            'perimeter': item.perimeter,
            'perimeter_crofton': item.perimeter_crofton,
            'solidity': item.solidity,
            'hu_1': item.moments_hu[0],
            'hu_2': item.moments_hu[1],
            'hu_3': item.moments_hu[2],
            'hu_4': item.moments_hu[3],
            'hu_5': item.moments_hu[4],
            'hu_6': item.moments_hu[5],
            'hu_7': item.moments_hu[6],
            'm_00': item.moments_central[0][0],
            'm_01': item.moments_central[0][1],
            'm_02': item.moments_central[0][2],
            'm_03': item.moments_central[0][3],
            'm_10': item.moments_central[1][0],
            'm_11': item.moments_central[1][1],
            'm_12': item.moments_central[1][2],
            'm_13': item.moments_central[1][3],
            'm_20': item.moments_central[2][0],
            'm_21': item.moments_central[2][1],
            'm_22': item.moments_central[2][2],
            'm_23': item.moments_central[2][3],
            'm_30': item.moments_central[3][0],
            'm_31': item.moments_central[3][1],
            'm_32': item.moments_central[3][2],
            'm_33': item.moments_central[3][3]
        }

        micro_list.append(micro_dict)

    img_to_haralick =  np.int16(img_array*4095)
    textures = mt.features.haralick(img_to_haralick, ignore_zeros=True, distance=1)

    #################### características dos ROIs
    roi_dict = {
        'roi_index': roi_label,
        'roi_result': roi_result,
        'n_obj': r_num, ### número de objetos no ROI
        'i_max': img_array[img_array != 0].max(),
        'i_mean': img_array[img_array != 0].mean(),
        'i_min': img_array[img_array != 0].min(),
        'i_std': img_array[img_array != 0].std(),
        'i_mean_min': img_array[img_array != 0].mean() - img_array[img_array != 0].min(),
        'i_mean_max': abs(img_array[img_array != 0].mean() - img_array[img_array != 0].max()),
        'i_max_min': img_array[img_array != 0].max() - img_array[img_array != 0].min(),
        'i_skew': stats.skew(img_array[img_array != 0]),
        'i_kurt': stats.kurtosis(img_array[img_array != 0]),
        'i_mode': stats.mode(img_array[img_array != 0])[0],
        'i_mode_count': stats.mode(img_array[img_array != 0])[1],
        'i_above_mode': np.count_nonzero(img_array[img_array != 0] > stats.mode(img_array[img_array != 0])[0]),
        'i_below_mode': np.count_nonzero(img_array[img_array != 0] < stats.mode(img_array[img_array != 0])[0]),
        't_ASM_0': textures[0][0],
        't_ASM_90': textures[1][0],
        't_ASM_180': textures[2][0],
        't_ASM_270': textures[3][0],
        't_contrast_0': textures[0][1],
        't_contrast_90': textures[1][1],
        't_contrast_180': textures[2][1],
        't_contrast_270': textures[3][1],
        't_correlation_0': textures[0][2],
        't_correlation_90': textures[1][2],
        't_correlation_180': textures[2][2],
        't_correlation_270': textures[3][2],
        't_sumSqrVariance_0': textures[0][3],
        't_sumSqrVariance_90': textures[1][3],
        't_sumSqrVariance_180': textures[2][3],
        't_sumSqrVariance_270': textures[3][3],
        't_idm_0': textures[0][4],
        't_idm_90': textures[1][4],
        't_idm_180': textures[2][4],
        't_idm_270': textures[3][4],
        't_sumAverage_0': textures[0][5],
        't_sumAverage_90': textures[1][5],
        't_sumAverage_180': textures[2][5],
        't_sumAverage_270': textures[3][5],
        't_sumVariance_0': textures[0][6],
        't_sumVariance_90': textures[1][6],
        't_sumVariance_180': textures[2][6],
        't_sumVariance_270': textures[3][6],
        't_sumEntropy_0': textures[0][7],
        't_sumEntropy_90': textures[1][7],
        't_sumEntropy_180': textures[2][7],
        't_sumEntropy_270': textures[3][7],
        't_entropy_0': textures[0][8],
        't_entropy_90': textures[1][8],
        't_entropy_180': textures[2][8],
        't_entropy_270': textures[3][8],
        't_diffVariance_0': textures[0][9],
        't_diffVariance_90': textures[1][9],
        't_diffVariance_180': textures[2][9],
        't_diffVariance_270': textures[3][9],
        't_diffEntropy_0': textures[0][10],
        't_diffEntropy_90': textures[1][10],
        't_diffEntropy_180': textures[2][10],
        't_diffEntropy_270': textures[3][10],
        't_IMC1_0': textures[0][11],
        't_IMC1_90': textures[1][11],
        't_IMC1_180': textures[2][11],
        't_IMC1_270': textures[3][11],
        't_IMC2_0': textures[0][12],
        't_IMC2_90': textures[1][12],
        't_IMC2_180': textures[2][12],
        't_IMC2_270': textures[3][12]
    }
    
    return (micro_list, roi_dict)

img_path = filedialog.askdirectory(title='Diretório das imagens originais')
gt_path = filedialog.askdirectory(title='Diretório do ground truth')
rst_path = filedialog.askdirectory(title='Diretório do resultados\objetos')

for item in os.listdir(rst_path):
    st = time.time() # get start time

    img_name = item[0:8]

    rst_array = io.imread(os.path.join(rst_path, item))
    
    gt_name = [data_name for data_name in os.listdir(gt_path) if str(img_name) in data_name]
    gt_array = io.imread(os.path.join(gt_path, gt_name[0]))

    ogn_name = [data_name for data_name in os.listdir(img_path) if str(img_name) in data_name]
    dicom_file = pydicom.dcmread(os.path.join(img_path, ogn_name[0]))
    ogn_array = dicom_file.pixel_array
    img_array = ogn_array/4095 # normaliza a imagem

    ##### (img_original, gt_array, resultado_watershed, nome da imagem)
    #### cada ROI tem um index (label), posição x e y, array original, do GT e do resultado correspondente
    roi_data = get_roi(img_array, gt_array, rst_array, ogn_name[0]) 

    img_roi_table = [] ###### tabela de caracteristicas por ROI
    img_micro_table = [] ###### tabela de caracteristicas das micros por ROI
    for roi in roi_data:

        try:
            ###### checar o resultado -> r_label e ground_truth
            if (roi['r_label'].max() != 0) and (roi['ground_truth'].max() == 0): ##### resultado TRUE e ground truth FALSE
                micro_list, roi_dict = get_features(roi['r_label'], roi['array'], roi['label'], 'false_positive')

                img_roi_table.append(roi_dict)
                img_micro_table.append(micro_list)

            elif (roi['r_label'].max() == 0) and (roi['ground_truth'].max() == 0):
                img_roi_table.append({'roi_index': roi['label'], 'roi_result': 'true_negative'})
            
            elif (roi['r_label'].max() != 0) and (roi['ground_truth'].max() != 0):
                micro_list, roi_dict = get_features(roi['r_label'], roi['array'], roi['label'], 'true_positive')
                
                img_roi_table.append(roi_dict)
                img_micro_table.append(micro_list)

            elif (roi['r_label'].max() == 0) and (roi['ground_truth'].max() != 0):
                img_roi_table.append({'roi_index': roi['label'], 'roi_result': 'false_negative'})

        except:
            print('ERROR img: '+str(img_name)+' ROI number ', roi['label'])
    
    df_roi_table = pd.DataFrame(img_roi_table)
    df_roi_table.to_csv(str(img_name)+'_roiFts.csv')

    final_micro_table = []
    for item_list in img_micro_table:
        [final_micro_table.append(item) for item in item_list]

    df_micro_table = pd.DataFrame(final_micro_table)
    df_micro_table.to_csv(str(img_name)+'_microFts.csv')

    et = time.time()
    et_final = et - st
    print('END: '+str(img_name)+' in ', et_final)

