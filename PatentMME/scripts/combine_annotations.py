import tqdm, json
import glob, os

if __name__ == '__main__':
    
    all_ann_folders = os.listdir('annotations')

    for folder in all_ann_folders:

        all_ann = glob.glob(f'annotations/{folder}/*.json')

        combined_ann = {}

        for ann in tqdm.tqdm(all_ann):
            with open(ann) as f:
                data = json.load(f)
            combined_ann.update(data)
        
        
        with open(f'annotations/combined_{folder}.json', 'w') as f:
            json.dump(combined_ann, f)