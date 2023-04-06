@echo off
set global_path=../../../vision_datasets
set data_dir=%global_path%

mkdir %data_dir%
cd /d %data_dir%

git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

echo "Downloading train and validation images"

powershell -Command "(New-Object System.Net.WebClient).DownloadFile('http://images.cocodataset.org/zips/train2017.zip', 'train2017.zip')"
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('http://images.cocodataset.org/zips/val2017.zip', 'val2017.zip')"

echo "Unzipping train folder"
powershell Expand-Archive -Path train2017.zip -DestinationPath .

echo "Unzipping val folder"
powershell Expand-Archive -Path val2017.zip -DestinationPath .

echo "Deleting zip files"
del /Q train2017.zip
del /Q val2017.zip

echo "COCO data downloading over!!"

cd ..
echo "Downloading annotations"
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', 'annotations_trainval2017.zip')"
powershell Expand-Archive -Path annotations_trainval2017.zip -DestinationPath .
del /Q annotations_trainval2017.zip
echo "Done"
