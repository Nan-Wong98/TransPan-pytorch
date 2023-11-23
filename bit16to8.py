import scipy.io as scio
import numpy as np
import cv2

class Bit16to8():
    def __init__(self,array,output_file,bands):
        self.array=array
        self.output_file=output_file
        self.bands=bands
        self.compress(array)

    def cumulativehistogram(self,array, band_min, band_max):
        """
        累计直方图统计
        Inputs:
        array_data:16位图像数据
        band_min:最小像素值
        band_max:最大像素值
        Outputs：
        cutmax:累积分布函数（CDF）==0.98对应的灰度级
        cutmin:累积分布函数（CDF）==0.02对应的灰度级
        """

        # 逐波段统计最值
        rows=array.shape[0]
        cols=array.shape[1]

        gray_level = int(band_max - band_min + 1)
        gray_array = np.zeros(gray_level)

        for row in range(rows):
            for col in range(cols):
                gray_array[int(array[row, col] - band_min)] += 1

        count_percent2 = rows*cols * 0.02
        count_percent98 = rows*cols* 0.98

        cutmax = 0
        cutmin = 0

        for i in range(1, gray_level):
            gray_array[i] += gray_array[i - 1]
            if (gray_array[i] >= count_percent2 and gray_array[i - 1] <= count_percent2):
                cutmin = i + band_min

            if (gray_array[i] >= count_percent98 and gray_array[i - 1] <= count_percent98):
                cutmax = i + band_min

        return cutmin, cutmax
    def compress(self,array):
        """
        Input:
        origin_16:16位图像路径
        Output:
        output:8位图像路径
        """
        rows,cols,channels=array.shape

        compress_data = np.zeros(( rows, cols,channels))

        for i in range(channels):
            band_max = np.max(array[:, :,i])
            band_min = np.min(array[:, :,i])

            cutmin, cutmax = self.cumulativehistogram(array[:, :,i], band_min, band_max)

            compress_scale = (cutmax - cutmin) / 255.0

            for j in range(rows):
                for k in range(cols):
                    if (array[j, k,i] < cutmin):
                        array[j, k,i] = cutmin

                    if (array[j, k,i] > cutmax):
                        array[j, k,i] = cutmax
                    compress_data[j, k,i] = (array[j, k,i] - cutmin) / compress_scale
        compress_data=compress_data.astype(np.uint8)
        self.save2tif(compress_data,self.bands,self.output_file)

    def save2tif(self,array,bands,output_filename):
        if bands==4:
            img_rgb=array[:,:,:3]
            cv2.imwrite(output_filename, img_rgb)
        elif bands==8:
            img_rgb=np.concatenate((array[:,:,1][:,:,np.newaxis],array[:,:,2][:,:,np.newaxis],array[:,:,4][:,:,np.newaxis]),axis=2)
            cv2.imwrite(output_filename, img_rgb)
        else:
            cv2.imwrite(output_filename,array)

