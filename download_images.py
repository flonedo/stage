import csv
import requests
import os


# downloads dataset images
def download_images(lst_dataset_file, output_path):
    # creates a new folder to save the downloaded images if it doesn't already exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # reads the content of the lst file
    with open(lst_dataset_file) as dataset:
        dataset_rows = csv.reader(dataset, delimiter='\t', quotechar='"')
        for row in dataset_rows:
            image_id = row[0]
            # the name of the downloaded images follows the format classname_index.jpg
            filename = image_id + '.jpg'
            url = row[10]
            result = requests.get(url, stream=True)
            if result.status_code == 200:
                image = result.raw.read()
                open(output_path + filename, "wb").write(image)


if __name__ == '__main__':
    # lst file containing the dataset details. It's readable as a tab separated csv
    lst_file = 'output/clientId/clientId.lst'
    output_path = r'download_images/'
    download_images(lst_file, output_path)
