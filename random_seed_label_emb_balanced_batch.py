import tensorflow as tf
import pandas as pd
from transformers import BertForSequenceClassification
import torch
from transformers import AutoModel,AutoTokenizer
import numpy as np
import json,os
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import time,datetime
from sklearn.metrics import classification_report
import random
from sampler import BalancedBatchSampler

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
def preprocess_data(tokenizer, sentences, MAX_LEN = 256):
    """
    :params[in]: tokenizer, the configured tokenizer
    :params[in]: sentences, list of strings
    """
    # 1. Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )        
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    
    # Set the maximum sequence length.
    # maximum training sentence length of 87...
    
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
    
    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return input_ids, attention_masks

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_eval(clf_model, train_meta, validation_dataloader, base_dir, batch_size,
               weights=None, lr=2e-5, epochs=4, eval_every_num_iters=40, seed_val = 42):
    """train and evaluate a deep learning model
    :params[in]: clf_model, a classifier
    :params[in]: train_meta, training data: data in ids, masks, and labels
    :params[in]: validation_dataloader, validation data
    :params[in]: base_dir, output directory to create the directory to save results
    :params[in]: lr, the learning rate
    :params[in]: epochs, the number of training epochs
    :params[in]: eval_every_num_iters, the number of iterations to evaluate
    :params[in]: seed_val, set a random seed
    """
    # the 'W' stands for 'Warm up", AdamW is a class from the huggingface library
    optimizer = AdamW(clf_model.parameters(),
                      lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = epochs
    train_inputs, train_masks, train_labels = train_meta
    train_size = train_inputs.shape[0]   # training sample size
    # Total number of training steps is number of batches * number of epochs.
    total_steps = int(1.+train_size/batch_size) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 1, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    # see if weights is None:
    if weights != None:
        weights = torch.FloatTensor(weights)
    # Set the seed value all over the place to make this reproducible.
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    
    ## reconstruct a dataloader
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        perm0 = torch.randperm(train_size)
        tmp_X,tmp_mask,tmp_Y = train_inputs[perm0,:], train_masks[perm0,:], train_labels[perm0]
        dataset = torch.utils.data.TensorDataset(tmp_X,tmp_mask,tmp_Y)
        train_loader = torch.utils.data.DataLoader(dataset,
                                        sampler=BalancedBatchSampler(dataset,tmp_Y),
                                        batch_size=batch_size, drop_last=True)
        # Measure how long the training epoch takes.
        t0 = time.time()
    
        # Reset the total loss for this epoch.
        total_loss = 0
    
        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        clf_model.train()  ## model training mode
    
        # For each batch of training data...
        for step, batch in enumerate(train_loader):
    
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
    
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            clf_model.zero_grad()        
    
            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = clf_model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels,
                        weights=weights)
                        #weights=torch.FloatTensor([100/127,100/191,100/34]))
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]
    
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()
    
            # Perform a backward pass to calculate the gradients.
            loss.backward()
    
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(clf_model.parameters(), 1.0)
    
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()    
            # Update the learning rate.
            scheduler.step()
            # eveluate the performance after some iterations
            if step % eval_every_num_iters == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))
                tmp_dir = base_dir+'/epoch'+str(epoch_i+1)+'iteration'+str(step)
                ## save pretrained model
                evaluate_model(clf_model, validation_dataloader, tmp_dir)
                clf_model.train()  ## model training mode
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / total_steps            
        
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        # save the data after epochs
        tmp_dir = base_dir+'/epoch'+str(epoch_i+1)+'_done'
        ## save pretrained model
        evaluate_model(clf_model, validation_dataloader, tmp_dir)
        clf_model.train()  ## model training mode

### evaluate the performance of current model
def evaluate_model(clf_model, validation_dataloader, save_dir):
    """
    :params[in]: clf_model, the pre-trained classifier
    :params[in]: validation_dataloader, the validation dataset
    :params[in]: save_dir, the directory name to save the fine-tuned model
    
    """
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    clf_model.eval()
    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    true_labels,pred_labels=[],[]
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = clf_model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        ## pred_labels/true_labels in a batch flatten
        pred_flat = np.argmax(logits, axis=1).flatten()
        true_flat = label_ids.flatten()

        # true labels and predicted labels
        true_labels += true_flat.tolist()
        pred_labels += pred_flat.tolist()
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    clf_model.save_pretrained(save_dir)  ## save model
    print(classification_report(true_labels, pred_labels,digits=3),
      file=open(save_dir+'/result.txt','a'))
    print("  Accuracy: {0:.3f}".format(eval_accuracy/nb_eval_steps),
          file=open(save_dir+'/result.txt','a'))
    
if __name__=='__main__':
    from transformers import BertForSequenceClassification_label
    from sklearn.model_selection import train_test_split
    tokenizer = AutoTokenizer.from_pretrained("biobert_v1.1_pubmed", from_tf=True)
    ## set the padding
    tokenizer.pad_token = '[PAD]'
    # tokenizer.pad_token_id
    data0 = json.load(open('original_labelled_data.json', 'r'))
    sentences,labels = data0['sentences'],data0['labels']
    #lm_model = AutoModelWithLMHead.from_pretrained("biobert_v1.1_pubmed", from_tf=True)
    ## loop over train/test data splits
    for seed in [42, 52, 62, 72, 82]:
        ## re-initialize the model for each seed
        base_dir = 'random_seeds/bio_bert_label_emb'+str(seed)
        x_train, x_test, y_train, y_test = train_test_split(sentences, labels,\
                          random_state=2020, stratify=labels, test_size=0.2)
        ## use preprocess_date function
        train_inputs,train_masks = preprocess_data(tokenizer, x_train, MAX_LEN = 256)
        validation_inputs,validation_masks = preprocess_data(tokenizer, x_test, MAX_LEN = 256)
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(y_train)
        validation_labels = torch.tensor(y_test)
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)
        ## meta data for training
        train_meta = (train_inputs, train_masks, train_labels)
        ## initialize classifier
        clf_model=BertForSequenceClassification_label.from_pretrained(
            "biobert_v1.1_pubmed",
            num_labels=len(set(labels)),
            from_tf=True)
        # For fine-tuning BERT on a specific task, the authors recommend a batch size of
        # 16 or 32.
        ## to make balanced batches, use multiples of number of classes as batch_size
        batch_size = 6   
        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data,  sampler=validation_sampler,batch_size=batch_size)
        train_eval(clf_model, train_meta, validation_dataloader, base_dir,batch_size=batch_size,\
            lr=2e-5, epochs=4, eval_every_num_iters=160, seed_val = seed)
        #, weights=[100/127,100/191,100/34])
