
import os
import sys
import getopt
import ijson
import wget

from zipfile import ZipFile

coco_labels=[]

def download_url():
    '''
    Obtain the download URL for the annotations based on the dataset release year
    '''
    url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


def json_file_path(basedir='/tmp'):
    '''
    Check if the JSON file exists in the designated directory
    '''
    return os.path.join(basedir,'annotations','instances_val2017.json')



def extract_annotations(target_dir='/tmp'):
    '''
    Extract the annotation zip file based on the dataset release year
    '''
    print('Downloading and extracting COCO annotations archive file')

    try:
        url = download_url()
        dest_zip_file = os.path.join(target_dir, 2017 + '.zip')
        wget.download(url,dest_zip_file)

        with ZipFile(dest_zip_file, 'r') as zip_archive:
            zip_archive.extractall(target_dir)

    except ValueError:
        print('Invalid value supplied to argument')

    except:
        print('Failed to extract zipped annotations')
        raise


def set_coco_labels():
    year = 2017
    json_file = json_file_path()

    if json_file is not None:
        if not os.path.isfile(json_file):
            extract_annotations(year)

        fd = open(json_file,'r')
        objs = ijson.items(fd, 'categories.item')
        labels = (o for o in objs)
        count = 0
        for label in labels:
            print('id:{}, category:{}, super category:{}'.format(label['id'], label['name'], label['supercategory']))
            count += 1
            #coco_labels[count] = f"{label['id'],label['name'],label['supercategory']}"
            coco_labels.append( f"{label['name'],label['supercategory']}")

            
        print('Total categories/labels: ', count)

        fd.close()

def main():
    set_coco_labels()
    print("Hello from main() function!")
    print(coco_labels)

if __name__ == "__main__":
    main()