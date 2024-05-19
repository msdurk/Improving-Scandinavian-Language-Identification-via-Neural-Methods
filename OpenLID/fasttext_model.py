import fasttext

def train_model(input_path, output_path):
    model = fasttext.train_supervised(
        input=input_path,
        lr=0.8,               
        dim=256,             
        epoch=2,             
        minCount=1000,      
        minn=2,              
        maxn=5,             
        wordNgrams=1,        
        bucket=1000000,     
        thread=68,           
        loss='softmax'       
    )
    model.save_model(output_path)

input_file = '/cluster/work/projects/ec30/ec-victocla/silver_large.txt'
output_model = '/cluster/work/projects/ec30/ec-victocla/fasttext_model_with_silver_big.bin'

train_model(input_file, output_model)
