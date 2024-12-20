import json
import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    frcnn_scaled_anns = '/DATA/nakul/PatentQA/main_methodology/LayoutLMv3/annotations/combined_scaled_annotations.json'

    with open(frcnn_scaled_anns, 'r') as f:
        block_annotations = json.load(f)
    block_annotations = dict(sorted(block_annotations.items()))

    node_counts = []
    with open('/DATA/nakul/PatentQA/main_methodology/LayoutLMv3/annotations/count_nodes.txt', 'a') as f:
        for im_id, im_name in tqdm.tqdm(enumerate(block_annotations)):
            im_blck_anns = block_annotations[im_name]
            
            count = 0
            for ctg_anns in im_blck_anns:
                if ctg_anns['category_id'] == 0:
                    count += 1
            
            node_counts.append(count)
            f.write(str(count) + '\n')
    
    print(sum(node_counts)/len(node_counts))

    plt.hist(node_counts)
    plt.savefig("/DATA/nakul/PatentQA/main_methodology/LayoutLMv3/annotations/node_count.jpg")
