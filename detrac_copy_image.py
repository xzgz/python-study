import os
import shutil

# xml路径的地址
XmlPath = r'/media/xzgz/ubudata/Ubuntu/Data/UA-DETRAC/DETRAC-train-xml'
# 原图片的地址
pictureBasePath = r"/media/xzgz/ubudata/Ubuntu/Data/UA-DETRAC/Insight-MVT_Annotation_Train"
# 保存图片的地址
saveBasePath = r"/media/xzgz/ubudata/Ubuntu/Data/UA-DETRAC/DETRAC-voc/JPEGImages"

total_xml = os.listdir(XmlPath)
num = len(total_xml)
list = range(num)
if os.path.exists(saveBasePath) == False:  # 判断文件夹是否存在
    os.makedirs(saveBasePath)

i = 0
for xml in total_xml:
    xml_temp = xml.split("__")
    folder = xml_temp[0]
    filename = xml_temp[1].split(".")[0] + ".jpg"
    # print(folder)
    # print(filename)
    temp_pictureBasePath = os.path.join(pictureBasePath, folder)
    filePath = os.path.join(temp_pictureBasePath, filename)
    # print(filePath)
    newfile = xml.split(".")[0] + ".jpg"
    newfile_path = os.path.join(saveBasePath, newfile)
    # print(newfile_path)
    shutil.copyfile(filePath, newfile_path)
    i += 1
    print(i)
print("xml file total number", num)
