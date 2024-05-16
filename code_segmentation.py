import os
import pydicom
import pywt
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tkinter import filedialog
from scipy import signal
from skimage import morphology, measure, segmentation, filters, util, io

def wavelet_filter(img_array, wave_name): # in: image float64, out img float64
    img = np.copy(img_array)

    k = 3
    cA, (cH, cV, cD) = pywt.wavedec2(img, wavelet=wave_name, level=1) # decompondo a imagem -> wavelet

    ####### mostrar os componentes
    # c_detail = cH + cV + cD
    # titles = ['Approximation', ' Horizontal detail',
    #       'Vertical detail', 'Diagonal detail']
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([cA, cH, cV, cD]):
    #     ax = fig.add_subplot(1, 4, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #     ax.set_title(titles[i], fontsize=10)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # fig.tight_layout()
    # plt.show()

    cH_var = np.var(cH)
    cH1 = math.sqrt(cH_var)*math.sqrt(math.pi/2)
    cH2 = math.sqrt(((4-math.pi)/2)*cH_var)
    cH_t = cH1 + (k*cH2)
    cH_new = pywt.threshold(data=cH, value=cH_t, mode='soft', substitute=0)

    cV_var = np.var(cV)
    cV1 = math.sqrt(cV_var)*math.sqrt(math.pi/2)
    cV2 = math.sqrt(((4-math.pi)/2)*cV_var)
    cV_t = cV1 + (k*cV2)
    cV_new = pywt.threshold(data=cV, value=cV_t, mode='soft', substitute=0)

    cD_var = np.var(cD)
    cD1 = math.sqrt(cD_var)*math.sqrt(math.pi/2)
    cD2 = math.sqrt(((4-math.pi)/2)*cD_var)
    cD_t = cD1 + (k*cD2)
    cD_new = pywt.threshold(data=cD, value=cD_t, mode='soft', substitute=0)

    result = pywt.waverec2(coeffs=[cA, (cH_new, cV_new, cD_new)], wavelet=wave_name) # reconstruindo a imagem

    ######## mostrar componentes limiarizados
    # titles = ['Approximation', ' Horizontal detail',
    #       'Vertical detail', 'Diagonal detail']
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([cA, cH_new, cV_new, cD_new]):
    #     ax = fig.add_subplot(1, 4, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #     ax.set_title(titles[i], fontsize=10)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # fig.tight_layout()
    # plt.show()

    ######### mostrar imagem reconstruida
    # fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis('off')
    # ax[1].imshow(result*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[1].axis('off')
    # fig.tight_layout()
    # plt.show()

    result[result < 0] = 0
    result_2 = result*4095
    result_3 = result_2.astype('int')
    result_4 = result_3/4095

    return (result_4) # para visualizar a imagem precisa passar para ubit16

def segm_1(img_array, h_value, se_type, thresh_method):
    img = np.copy(img_array)

    ##### h_maxima
    h_maxima = morphology.reconstruction(img-h_value, img, footprint=se_type, method='dilation')
    h_convex = img - h_maxima

    ##### Threshold de h_convex
    ###### O número inteiro é o número máximo de regiões no threshold -> Threshold iterativo
    if isinstance(thresh_method, int): 
        for i in np.arange(h_convex.max(), 0, -0.0003):
            im_segm = h_convex > i
            seg_label, seg_num = measure.label(im_segm, return_num=True, connectivity=2)
            # seg_props = measure.regionprops_table(label_image=seg_label, intensity_image=img, properties=['label', 'area', 'intensity_max'])
            # seg_df = pd.DataFrame(seg_props)

            if seg_num < thresh_method:
                thresh = i
            else:
                break
        
        marker_1 = h_convex > thresh
    
    ###### Limiarização por Otsu
    elif thresh_method == 'otsu':
        thresh = filters.threshold_otsu(h_convex)
        marker_1 = h_convex > thresh
        marker_1 = morphology.area_opening(marker_1, area_threshold=5)

    ##### Todos os pixels diferentes de 0
    elif thresh_method == 'all':
        marker_1 = h_convex!=0
        marker_1 = morphology.area_opening(marker_1, area_threshold=5)

    ##### Selecionando os marcadores
    m_labels, m_num = measure.label(marker_1, return_num=True, connectivity=2)
    m_props = measure.regionprops_table(m_labels, img, properties=['label', 'area_bbox'])
    m_props = pd.DataFrame(m_props, index=m_props['label'])

    reject_list = m_props.index[(m_props['area_bbox'] > 250)].tolist()
    reject_list = list(dict.fromkeys(reject_list))

    selected_markers = np.copy(m_labels)
    for label in reject_list:
        selected_markers[selected_markers == label] = 0

    int_marker = selected_markers!=0

    ##### Marcador externo
    ### watershed com a imagem filtrada e invertida -> closing-opening da imagem original
    im_c = morphology.closing(img, footprint=np.ones((3,3)))
    im_co = morphology.opening(im_c, footprint=np.ones((3,3)))

    im_comp = util.invert(im_co)
    i_marker_label = measure.label(int_marker, connectivity=2)
    im_co_ws = segmentation.watershed(im_comp, watershed_line=True, markers=i_marker_label)
    ext_marker = im_co_ws == 0

    ext_marker_d = morphology.dilation(ext_marker, footprint=np.ones((3,3)))
    int_marker_2 = int_marker.astype('int')-ext_marker_d.astype('int')
    int_marker_3 = int_marker_2  == 1

    final_marker = ext_marker + int_marker_3

    ###### Segmentação por watershed
    img_grad = morphology.dilation(img, footprint=np.ones((3,3))) - morphology.erosion(img, footprint=np.ones((3,3)))

    r_markers = measure.label(final_marker, connectivity=2)
    r_watershed = segmentation.watershed(img_grad, markers=r_markers)

    ###### Selecionando as regiões segmentadas
    w_props = measure.regionprops_table(r_watershed, img, properties=['label', 'area_bbox'])
    w_props = pd.DataFrame(w_props, index=w_props['label'])

    w_reject_list = w_props.index[(w_props['area_bbox'] > 250)].tolist()
    w_reject_list = w_reject_list + (w_props.index[(w_props['area_bbox'] < 5)].tolist()) 
    w_reject_list = list(dict.fromkeys(w_reject_list))

    new_w = np.copy(r_watershed)
    for label in w_reject_list:
        new_w[new_w == label] = 0

    ###### PLOTS
    #### H-maxima e H-convex
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,30), sharex=True, sharey=True)
    # ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis('off')
    # ax[0].set_title('original')
    # ax[1].imshow(h_maxima, cmap='gray', vmin=0)
    # ax[1].axis('off')
    # ax[1].set_title('h-maxima')
    # ax[2].imshow(h_convex, cmap='gray', vmin=0)
    # ax[2].axis('off')
    # ax[2].set_title('h-convex')
    # plt.show()

    ##### Marcadores extraídos de h_maxima
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,30), sharex=True, sharey=True)
    # ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis('off')
    # ax[0].set_title('original')
    # ax[1].imshow(segmentation.mark_boundaries(img, mark_1, mode='outer'), cmap='gray', vmin=0)
    # ax[1].axis('off')
    # ax[1].set_title('h-maxima markers')
    # plt.show()

    ##### Marcador interno
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,30), sharex=True, sharey=True)
    # ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis('off')
    # ax[0].set_title('original')
    # ax[1].imshow(segmentation.mark_boundaries(img, marker_1, mode='outer'), cmap='gray', vmin=0)
    # ax[1].axis('off')
    # ax[1].set_title('Resultado do threshold')
    # ax[2].imshow(segmentation.mark_boundaries(img, int_marker, mode='outer'), cmap='gray', vmin=0)
    # ax[2].axis('off')
    # ax[2].set_title('Marcadores internos')
    # plt.show()

    ##### Marcadores para segmentação
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,30), sharex=True, sharey=True)
    # ax[0,0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0,0].axis('off')
    # ax[0,0].set_title('original')
    # ax[0,1].imshow(segmentation.mark_boundaries(img, morphology.dilation(int_marker, footprint=np.ones((3,3))), mode='outer'), cmap='gray', vmin=0)
    # ax[0,1].axis('off')
    # ax[0,1].set_title('int marker')
    # ax[1,0].imshow(segmentation.mark_boundaries(img, ext_marker, mode='outer'), cmap='gray', vmin=0)
    # ax[1,0].axis('off')
    # ax[1,0].set_title('ext marker')
    # ax[1,1].imshow(segmentation.mark_boundaries(img, final_marker, mode='outer'), cmap='gray', vmin=0)
    # ax[1,1].axis('off')
    # ax[1,1].set_title('final marker')
    # plt.show()

    ##### Regiões segmentadas
    # plt.imshow(segmentation.mark_boundaries(img, r_watershed, mode='outer'))
    # plt.axis('off')
    # plt.title('result - final markers')
    # plt.show()

    ##### Resultado final
    # plt.imshow(segmentation.mark_boundaries(img, new_w, mode='outer'))
    # plt.axis('off')
    # plt.title('result - selected regions')
    # plt.show()

    return (int_marker, r_watershed, new_w)

def segm_2(img_array, se_type, thresh_method):
    img = np.copy(img_array)

    ##### top hat com reconstrução
    img_e = morphology.erosion(img, footprint=se_type)
    img_obr = morphology.reconstruction(img_e, img, method='dilation', footprint=se_type)
    img_th = img - img_obr

    ##### Threshold de top-hat
    ###### O número inteiro é o número máximo de regiões no threshold -> Threshold iterativo
    if isinstance(thresh_method, int): 
        for i in np.arange(img_th.max(), 0, -0.0003):
            im_segm = img_th > i
            seg_label, seg_num = measure.label(im_segm, return_num=True, connectivity=2)
            # seg_props = measure.regionprops_table(label_image=seg_label, intensity_image=img, properties=['label', 'area', 'intensity_max'])
            # seg_df = pd.DataFrame(seg_props)

            if seg_num < thresh_method:
                thresh = i
            else:
                break
        
        marker_1 = img_th > thresh
    
    ###### Limiarização por Otsu
    elif thresh_method == 'otsu':
        thresh = filters.threshold_otsu(img_th)
        marker_1 = img_th > thresh
        marker_1 = morphology.area_opening(marker_1, area_threshold=5)

    ##### Todos os pixels diferentes de 0
    elif thresh_method == 'all':
        marker_1 = img_th!=0
        marker_1 = morphology.area_opening(marker_1, area_threshold=5)

    ##### Selecionando os marcadores
    m_labels, m_num = measure.label(marker_1, return_num=True, connectivity=2)
    m_props = measure.regionprops_table(m_labels, img, properties=['label', 'area_bbox'])
    m_props = pd.DataFrame(m_props, index=m_props['label'])

    reject_list = m_props.index[(m_props['area_bbox'] > 250)].tolist()
    reject_list = list(dict.fromkeys(reject_list))

    selected_markers = np.copy(m_labels)
    for label in reject_list:
        selected_markers[selected_markers == label] = 0

    int_marker = selected_markers!=0

    ##### Marcador externo
    ### watershed com a imagem filtrada e invertida -> closing-opening da imagem original
    im_c = morphology.closing(img, footprint=np.ones((3,3)))
    im_co = morphology.opening(im_c, footprint=np.ones((3,3)))

    im_comp = util.invert(im_co)
    i_marker_label = measure.label(int_marker, connectivity=2)
    im_co_ws = segmentation.watershed(im_comp, watershed_line=True, markers=i_marker_label)
    ext_marker = im_co_ws == 0

    ext_marker_d = morphology.dilation(ext_marker, footprint=np.ones((3,3)))
    int_marker_2 = int_marker.astype('int')-ext_marker_d.astype('int')
    int_marker_3 = int_marker_2  == 1

    final_marker = ext_marker + int_marker_3

    ###### Segmentação por watershed
    img_grad = morphology.dilation(img, footprint=np.ones((3,3))) - morphology.erosion(img, footprint=np.ones((3,3)))

    r_markers = measure.label(final_marker, connectivity=2)
    r_watershed = segmentation.watershed(img_grad, markers=r_markers)

    ###### Selecionando as regiões segmentadas
    w_props = measure.regionprops_table(r_watershed, img, properties=['label', 'area_bbox'])
    w_props = pd.DataFrame(w_props, index=w_props['label'])

    w_reject_list = w_props.index[(w_props['area_bbox'] > 250)].tolist()
    w_reject_list = w_reject_list + (w_props.index[(w_props['area_bbox'] < 5)].tolist()) 
    w_reject_list = list(dict.fromkeys(w_reject_list))

    new_w = np.copy(r_watershed)
    for label in w_reject_list:
        new_w[new_w == label] = 0

    return(int_marker, r_watershed, new_w)

img_path = filedialog.askdirectory(title='Diretório das imagens originiais')

for img_name in os.listdir(img_path):
    #### Carregar a imagem original
    dicom_file = pydicom.dcmread(os.path.join(img_path, img_name))
    original_array = dicom_file.pixel_array

    #### Normalizar a imagem original
    img_array =  original_array/4095

    #### Pré-processamento
    img_wavelet = wavelet_filter(img_array, 'coif5')
    img_wiener = signal.wiener(img_array, mysize=(3,3))

    #### Segmentação
    [wave_markers_1, wave_watershed_1, wave_rst_1] = segm_1(img_wavelet, 0.04, morphology.disk(11), 1000)
    [wave_markers_2, wave_watershed_2, wave_rst_2] = segm_2(img_wavelet, morphology.disk(11), 1000)

    [wiener_markers_1, wiener_watershed_1, wiener_rst_1] = segm_1(img_wiener, 0.04, morphology.disk(11), 1000)
    [wiener_markers_2, wiener_watershed_2, wiener_rst_2] = segm_2(img_wiener, morphology.disk(11), 1000)

    io.imsave(str(img_name[0:8])+'_wave_'+'markers_segm1.png', util.img_as_ubyte(wave_markers_1!=0))
    io.imsave(str(img_name[0:8])+'_wave_'+'rst_segm1.png', util.img_as_ubyte(wave_rst_1!=0))
    io.imsave(str(img_name[0:8])+'_wiener_'+'mark_seg1.png', util.img_as_ubyte(wiener_markers_1!=0))
    io.imsave(str(img_name[0:8])+'_wiener_'+'rst_seg1.png', util.img_as_ubyte(wiener_rst_1!=0))

    io.imsave(str(img_name[0:8])+'_wave_'+'markers_segm2.png', util.img_as_ubyte(wave_markers_2!=0))
    io.imsave(str(img_name[0:8])+'_wave_'+'rst_segm2.png', util.img_as_ubyte(wave_rst_2!=0))
    io.imsave(str(img_name[0:8])+'_wiener_'+'mark_seg2.png', util.img_as_ubyte(wiener_markers_2!=0))
    io.imsave(str(img_name[0:8])+'_wiener_'+'rst_seg2.png', util.img_as_ubyte(wiener_rst_2!=0))

    ####### PLOTS
    #### Plotar imagem original
    # plt.imshow(np.int16(img_array*4095), cmap='gray', vmin=0, vmax=4095)
    # plt.axis(False)
    # plt.show()

    #### Plotar imagens após pré-processamento
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,30), sharex=True, sharey=True)
    # ax[0].imshow(np.int16(img_array*4095), cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis(False)
    # ax[1].imshow(np.int16(img_wavelet*4095), cmap='gray', vmin=0, vmax=4095)
    # ax[1].axis(False)
    # ax[2].imshow(np.int16(img_wiener*4095), cmap='gray', vmin=0, vmax=4095)
    # ax[2].axis(False)
    # plt.show()

    print(img_name)

