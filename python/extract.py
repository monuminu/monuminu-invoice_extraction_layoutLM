import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import os
import numpy as np
import pandas as pd
import torch
from transformers import LayoutLMv2Tokenizer, LayoutLMv2ForTokenClassification
import pytesseract
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import ImageDraw, ImageFont, Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AdamW
from tqdm.notebook import tqdm
import numpy as np
import torch
from transformers import LayoutLMv2Tokenizer, LayoutLMv2ForTokenClassification, LayoutLMv2Config
config = LayoutLMv2Config.from_pretrained("/mnt/d/Data_Science_Work/hello-vue3/python/invoice_extraction_layoutlmv2")
tokenizer = LayoutLMv2Tokenizer.from_pretrained("/mnt/d/Data_Science_Work/hello-vue3/python/invoice_extraction_layoutlmv2")
model = LayoutLMv2ForTokenClassification.from_pretrained("/mnt/d/Data_Science_Work/hello-vue3/python/invoice_extraction_layoutlmv2", config = config)
model.to(device)


def normalize_box(box, width, height):
    width = int(width)
    height = int(height)
    return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]

def de_normalize(box, width, height):
  return [
          int((width * box[0])/1000),
          int((height * box[1])/1000),
          int((width * box[2])/1000),
          int((height * box[3])/1000)
  ]

def resize_and_align_bounding_box(bbox, original_image, target_size):
    x_, y_ = original_image.size
    x_scale = target_size / x_ 
    y_scale = target_size / y_
    origLeft, origTop, origRight, origBottom = tuple(bbox)
    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale)) 
    return [x-0.5, y-0.5, xmax+0.5, ymax+0.5]
 
class InvoiceDataSet(Dataset):
    """LayoutLM dataset with visual features."""
 
    def __init__(self, df, tokenizer, max_length, target_size, train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.target_size = target_size
        self.pad_token_box = [0, 0, 0, 0]
        self.train = train
 
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx,:].to_dict()        
        #base_path = data_config.base_image_path
        original_image = Image.open(item["imageFilename"]).convert("RGB")
        # resize to target size (to be provided to the pre-trained backbone)
        resized_image = original_image.resize((self.target_size, self.target_size))
        # first, read in annotations at word-level (words, bounding boxes, labels)
        words = item["words"]
        unnormalized_word_boxes = item["bbox"]
        word_labels = item["label"]
        width = item["imageWidth"]
        height = item["imageHeight"]
        normalized_word_boxes = [normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
        assert len(words) == len(normalized_word_boxes)
 
        # next, transform to token-level (input_ids, attention_mask, token_type_ids, bbox, labels)
        token_boxes = []
        unnormalized_token_boxes = []
        token_labels = []
        for word, unnormalized_box, box, label in zip(words, unnormalized_word_boxes, normalized_word_boxes, word_labels):
            word_tokens = self.tokenizer.tokenize(word)
            unnormalized_token_boxes.extend(unnormalized_box for _ in range(len(word_tokens)))
            token_boxes.extend(box for _ in range(len(word_tokens)))
            # label first token as B-label (beginning), label all remaining tokens as I-label (inside)
            for i in range(len(word_tokens)):
                if  1 == 1:#i == 0:
                    token_labels.extend(['B-' + label])
                else:
                    token_labels.extend(['I-' + label])
        
        # Truncation of token_boxes + token_labels
        special_tokens_count = 2 
        if len(token_boxes) > self.max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (self.max_seq_length - special_tokens_count)]
            unnormalized_token_boxes = unnormalized_token_boxes[: (self.max_seq_length - special_tokens_count)]
            token_labels = token_labels[: (self.max_seq_length - special_tokens_count)]
        
        # add bounding boxes and labels of cls + sep tokens
        token_boxes = [self.pad_token_box] + token_boxes + [[1000, 1000, 1000, 1000]]
        unnormalized_token_boxes = [self.pad_token_box] + unnormalized_token_boxes + [[1000, 1000, 1000, 1000]]
        token_labels = [-100] + token_labels + [-100]
        
        encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True)
        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = self.max_seq_length - len(input_ids)
        token_boxes += [self.pad_token_box] * padding_length
        unnormalized_token_boxes += [self.pad_token_box] * padding_length
        token_labels += [-100] * padding_length
        encoding['bbox'] = token_boxes
        encoding['labels'] = token_labels
 
        assert len(encoding['input_ids']) == self.max_seq_length
        assert len(encoding['attention_mask']) == self.max_seq_length
        assert len(encoding['token_type_ids']) == self.max_seq_length
        assert len(encoding['bbox']) == self.max_seq_length
        assert len(encoding['labels']) == self.max_seq_length
 
        encoding['resized_image'] = ToTensor()(resized_image)
        # rescale and align the bounding boxes to match the resized image size (typically 224x224) 
        encoding['resized_and_aligned_bounding_boxes'] = [resize_and_align_bounding_box(bbox, original_image, self.target_size) for bbox in unnormalized_token_boxes]
        #encoding['unnormalized_token_boxes'] = unnormalized_token_boxes
        
        # finally, convert everything to PyTorch tensors 
        for k,v in encoding.items():
            if k == 'labels':
                continue
                label_indices = []
                # convert labels from string to indices
                for label in encoding[k]:
                    if label != -100:
                        label_indices.append(data_config.label2id[label])
                    else:
                        label_indices.append(label)
                encoding[k] = label_indices
            encoding[k] = torch.as_tensor(encoding[k])
        return encoding

def get_ocr(image, width, height):
  w_scale = 1000/width
  h_scale = 1000/height
  config = r'--oem 3 --psm 6'
  ocr_df = pytesseract.image_to_data(image, output_type = 'data.frame', lang = 'eng', config = config)
  ocr_df = ocr_df.dropna().assign(left_scaled = ocr_df.left * w_scale,
                                  width_scaled = ocr_df.width*w_scale,
                                  top_scaled = ocr_df.top*h_scale,
                                  height_scaled = ocr_df.height*h_scale,
                                  right_scaled = lambda x : x.left_scaled + x.width_scaled)
  float_cols = ocr_df.select_dtypes('float').columns
  ocr_df = ocr_df.dropna().reset_index(drop=True)
  ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
  ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
  ocr_df = ocr_df.dropna().reset_index(drop=True)
  words = list(ocr_df.text)
  words = [str(w) for w in words]
  coordinates = ocr_df[['left', 'top', 'width', 'height']]
  actual_boxes = []
  for idx, row in coordinates.iterrows():
      x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
      actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+width, top+height) to get the actual box 
      actual_boxes.append(actual_box)
  return words, actual_boxes


def predict(pred_dataloader, model):
    model.eval()
    tk0 = tqdm(pred_dataloader, total = len(pred_dataloader))
    for bi, batch in enumerate(tk0):
        with torch.no_grad():
            input_ids=batch['input_ids'].to(device)
            bbox=batch['bbox'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            token_type_ids=batch['token_type_ids'].to(device)
            resized_images = batch['resized_image'].to(device) 
            resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(device)
            model_outputs = model(image = resized_images,input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
    token_boxes = bbox.squeeze().tolist()
    token_ids = input_ids.squeeze().tolist()
    token_predictions = model_outputs.logits.argmax(-1).squeeze().tolist()
    output = []
    for label, label_name in config.id2label.items():
      if label_name in ("B-other","I-other"):
        continue
      answer = tokenizer.decode([token_ids[i] for i, x in enumerate(token_predictions) if x == label])
      boxes = [token_boxes[i] for i, x in enumerate(token_predictions) if x == label]
      output.append({"label_name" : label_name, "answer" : answer, "boxes" : boxes})
    return output  
    
label2color = {'B-invoice_info': 'blue',
 'B-other': 'white',
 'B-positions': 'red',
 'B-receiver': 'violet',
 'B-supplier': 'orange',
 'B-total': 'green'}

def draw_image(imageFilename, df_out):
  image = Image.open(imageFilename)
  image = image.convert('RGB')
  width, height = image.size
  draw = ImageDraw.Draw(image)
  font = ImageFont.load_default()
  for i in range(df_out.shape[0]):
    boxes = df_out.loc[i, "boxes"]
    label_name = df_out.loc[i, "label_name"]
    for box in boxes:
      box = de_normalize(box, width, height)
      draw.rectangle(box, outline = label2color[label_name])
    draw.text((615 ,255 + (i * 20)), text = label_name,fill = label2color[label_name], font = font)
    draw.rectangle([550 ,250 + (i * 20),600 ,250 + ((i + 1) * 20)],fill = label2color[label_name])
  return image    


  
def fetch_invoice_details(imageFilename):  
  image = Image.open(imageFilename)
  image = image.convert('RGB')
  width, height = image.size
  words, boxes = get_ocr(image, width, height) 
  predict_df = pd.DataFrame({"imageFilename" : imageFilename	,"imageHeight" : height	,"imageWidth" : width	, "words" : [words]	, "bbox" : [boxes],	"label" : [["other"] * len(words)]})
  pred_dataset = InvoiceDataSet(df = predict_df, tokenizer = tokenizer, max_length = 512, target_size = 224, train=False)
  pred_dataloader = DataLoader(pred_dataset, batch_size=4)
  result = predict(pred_dataloader, model) 
  return result