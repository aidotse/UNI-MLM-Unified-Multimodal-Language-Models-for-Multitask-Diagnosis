import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from focal_loss.focal_loss import FocalLoss
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics





####
####
####
####                                                                                ARCHITECTURES FOR PROJECTION MODULES
####
####
####
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded






####
####
####
####                                                                                DATA-RELATED CODE
####                                                                        Includes a custom dataset class as well
####                                                                        as a preproccesing and data splitting class
####
####
####
class CustomDataset(Dataset):
    """ 
    CustomDataset is a PyTorch Dataset subclass for handling vectors and labels. 
        Each vector is divided into distinct features and returned as a dictionary 
        along with the corresponding label.

        Attributes:
            vectors (list): A list of vectors representing data points.
            labels (list): A list of labels for each vector.

        Methods:
            __init__(vectors, labels):
                Initializes the dataset with vectors and labels.
            __len__():
                Returns the number of samples in the dataset.
            __getitem__(index):
                Returns the feature dictionary and label for the sample at the given index.
    """
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, index):
        vector = torch.tensor(self.vectors[index]).float()

        vd = vector[:1024]
        vmd = vector[1024:2048]
        ts_ce = vector[2048:2147]
        ts_le = vector[2147:2389]
        ts_pe = vector[2389:2499]
        n_rad = vector[2499:]

        feature_dict = {'vd': vd, 'vmd': vmd, 'ts_ce': ts_ce, 'ts_le': ts_le, 'ts_pe': ts_pe, 'n_rad': n_rad}

        label = torch.from_numpy(np.array([self.labels[index]]))
        return feature_dict, label.squeeze()


class DataSplit():
    """
    DataSplit class for partitioning and splitting data into training and validation sets.

    Attributes:
        df (DataFrame): Input DataFrame containing the data.
        types (list): List of prefixes for different types of data columns.
        partition (str): Indicates the current partition state of the data.

    Methods:
        __init__(df):
            Initializes the DataSplit object with the given DataFrame.
        partitiondata(partition):
            Partitions and preprocesses the data based on the specified partition type.
        split_data(partition, validation_size=0.25, random_state=23):
            Splits the data into training and validation sets.
        get_data():
            Returns the training and validation data after splitting.

    Details:
        - The class processes data by partitioning based on mortality and length-of-stay criteria.
        - It splits the data into training and validation sets and normalizes the data.
        - Columns related to specific types are dropped, and outliers in 'n_rad' features are removed.
    """



    def __init__(self, df):
        self.df = df
        self.types = ['demo_', 'vd_', 'vp_', 'vmd_', 'vmp', 'ts_ce_', 'ts_le_', 'ts_pe_', 'n_rad_']
        self.partition = None

    def partitiondata(self, partition):
        self.pkl_list = []

        if partition == 'all':
            df_mor = self.df

            df_death_small48 = df_mor[((df_mor['img_length_of_stay'] < 48) & (df_mor['death_status'] == 1))]
            df_alive_big48 = df_mor[((df_mor['img_length_of_stay'] >= 48) & (df_mor['death_status'] == 0))]
            df_death_big48 = df_mor[((df_mor['img_length_of_stay'] >= 48) & (df_mor['death_status'] == 1))]
            df_alive_small48 = df_mor[((df_mor['img_length_of_stay'] < 48) & (df_mor['death_status'] == 0))]

            df_death_small48['y'] = 1
            df_alive_big48['y'] = 0
            df_death_big48['y'] = 0
            df_alive_small48['y'] = 0
            df_mor = pd.concat([df_death_small48, df_alive_big48, df_death_big48, df_alive_small48], axis = 0)

            df_los = self.df

            df_alive_small48 = df_los[((df_los['img_length_of_stay'] < 48) & (df_los['death_status'] == 0))]
            df_alive_big48 = df_los[((df_los['img_length_of_stay'] >= 48) & (df_los['death_status'] == 0))]
            df_death = df_los[(df_los['death_status'] == 1)]

            df_alive_small48['y'] = 1
            df_alive_big48['y'] = 0
            df_death['y'] = 0
            df_los = pd.concat([df_death_small48, df_alive_big48, df_death_big48, df_alive_small48], axis = 0)

            self.df['48-hour Mortality'] = df_mor['y']
            self.df['Length-of-Stay'] = df_los['y']

            self.df['y'] = self.df.apply(lambda row: [row['Fracture'], row['Lung Lesion'], row['Enlarged Cardiomediastinum'], 
                                          row['Consolidation'], row['Pneumonia'], row['Atelectasis'], row['Lung Opacity'], row['Pneumothorax'],
                                          row['Edema'], row['Cardiomegaly'], row['Length-of-Stay'], row['48-hour Mortality']], axis=1)

            self.df = self.df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 
                    'death_status', 'split', 'No Finding', 'Fracture', 'Lung Lesion', 'Enlarged Cardiomediastinum', 'Consolidation', 'Pneumonia', 
                    'Atelectasis', 'Lung Opacity', 'Lung Opacity', 'Pneumothorax', 'Edema', 'Cardiomegaly', 'Pleural Effusion', 
                    'Pleural Other', 'Support Devices', 'PerformedProcedureStepDescription', 'ViewPosition', 
                    '48-hour Mortality', 'Length-of-Stay'], axis = 1)
            
            self.df = self.df.drop(list(self.df.filter(regex='^de_')), axis = 1)
            self.df = self.df.drop(list(self.df.filter(regex='^vp_')), axis = 1)
            self.df = self.df.drop(list(self.df.filter(regex='^vmp_')), axis = 1)

            # Remove outliers from n_rad-features
            n_rad_columns = [col for col in self.df.columns if col.startswith('n_rad_')]
            self.df = self.df[~(self.df[n_rad_columns] > 10).any(axis=1)]
            self.df = self.df.dropna()


    def split_data(self, partition, validation_size=0.25, random_state=23):

        self.partition = partition

        self.partitiondata(partition)
        pkl_list = self.df['haim_id'].unique().tolist()

        # Split into training and test sets
        train_id, val_id = train_test_split(pkl_list, test_size=validation_size, random_state=random_state)

        train_idx = self.df[self.df['haim_id'].isin(train_id)]['haim_id'].tolist()
        validation_idx = self.df[self.df['haim_id'].isin(val_id)]['haim_id'].tolist()

        self.x_train = self.df[self.df['haim_id'].isin(train_idx)].drop(['y','haim_id'],axis=1)
        self.x_val = self.df[self.df['haim_id'].isin(validation_idx)].drop(['y','haim_id'],axis=1)
        
        # Normalize the data using the mean and standard deviation of the training set
        for column in self.x_train.columns:
            mean = self.x_train[column].mean()
            std = self.x_train[column].std()
            
            if std != 0:
                self.x_train[column] = (self.x_train[column] - mean) / std
                self.x_val[column] = (self.x_val[column] - mean) / std
            else:
                continue

        self.y_train = self.df[self.df['haim_id'].isin(train_idx)]['y']
        self.y_val = self.df[self.df['haim_id'].isin(validation_idx)]['y']


    def get_data(self):
        train_cols = self.x_train
        val_cols = self.x_val
        return train_cols, val_cols








####
####
####
####                                                                                CUSTOM ASYMMETRIC LOSS
####
####
####
class AsymmetricLoss(nn.Module):
    """
    AsymmetricLoss is a custom loss function for handling imbalanced classes 
    by applying different penalties for positive and negative samples.

    Attributes:
        gamma_neg (float): Parameter to control the negative loss.
        clip (float): Value to clip the negative sample logits.
        eps (float): Small epsilon value to avoid log(0).

    Methods:
        __init__(gamma_neg=0, clip=0.05, eps=1e-6):
            Initializes the loss function with the given parameters.
        forward(x, y):
            Computes the asymmetric loss between the input logits and labels.
            Parameters:
                x: Input logit tensor.
                y: Label tensor.
            Returns:
                - Mean loss value.
                - Delta_p: Difference between average positive and negative predictions.
    """


    def __init__(self, gamma_neg=0, clip=0.05, eps=1e-6):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        pm = torch.max(x_sigmoid - self.clip, torch.tensor(0.0, device=x.device))
        xs_neg = 1-pm

        l_pos = y * (1-xs_pos) * torch.log(xs_pos.clamp(min=self.eps))
        l_neg = (1-y) * torch.pow(pm, self.gamma_neg) * torch.log(xs_neg.clamp(min=self.eps))

        loss = l_pos + l_neg

        avg_pos = torch.mean(x_sigmoid[y == 1]) if (y == 1).sum() > 0 else torch.tensor(0.1, device=x.device)
        avg_neg = torch.mean(1 - x_sigmoid[y == 0]) if (y == 0).sum() > 0 else torch.tensor(0.1, device=x.device)
        
        delta_p = avg_pos - avg_neg

        return -loss.mean(), delta_p




####
####
####
####                                                                                HELPER FUNCTIONS
####
####
####
def logit_extractor(emb, llm):
    """
    Extracts and processes logits from a language model given input embeddings.

    Parameters:
        emb (tensor): Input embeddings for the language model.
        llm (model): Language model to generate outputs from embeddings.

    Returns:
        logits (tensor): Averaged logits for specified class labels.
    """

    outputs = llm(inputs_embeds=emb)
    class_labels = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 1101, 1201]
    logits = outputs['logits']
    logits = logits[:,-1:,class_labels].mean(dim=1).float()
    return logits



def custom_asym_loss(logits, labels, loss_fns):
    """
    Computes a custom asymmetric loss for each class using specified loss functions.

    Parameters:
        logits (tensor): Logits for each class.
        labels (tensor): Labels for each class.
        loss_fns (list): List of loss functions, one for each class.

    Returns:
        loss (float): Sum of losses for all classes.
        delta_ps (list): List of delta_p values for each class.

    Details:
        - Splits the logits and labels into individual class tensors.
        - Masks out NaN and class '2' labels.
        - Applies the corresponding loss function to the masked logits and labels.
        - Aggregates the losses and delta_p values.
    """


    losses = []
    delta_ps = []
    class_tensors = torch.split(logits, 1, dim=1)
    class_labels = torch.split(labels, 1, dim=1)
    for i, fn in enumerate(loss_fns):
        mask = ~torch.isnan(class_labels[i]) & (class_labels[i] != 2)
        masked_labels = class_labels[i][mask]
        masked_logits = class_tensors[i][mask]
        if masked_labels.size(0) == 0:
            losses.append(0.0)
            delta_ps.append(0.1)
        else:
            loss, delta_p= fn(masked_logits, masked_labels) 
            losses.append(loss)
            delta_ps.append(delta_p)

    loss = sum(losses)

    return loss, delta_ps



def custom_mse_loss(decoded, inputs, mse_loss):
    """
    Computes custom MSE loss for each type of input and decoded output.

    Parameters:
        decoded (list): List of decoded outputs from the projectors.
        inputs (list): List of original input tensors.
        mse_loss (function): Mean Squared Error loss function.

    Returns:
        mse (list): List of MSE losses for each input-output pair.
    """

    vd_loss = mse_loss(decoded[0], inputs[0])
    vmd_loss = mse_loss(decoded[1], inputs[1])
    ts_pe_loss = mse_loss(decoded[2], inputs[2])
    ts_ce_loss = mse_loss(decoded[3], inputs[3])
    ts_le_loss = mse_loss(decoded[4], inputs[4])
    n_rad_loss = mse_loss(decoded[5], inputs[5])

    mse = [vd_loss, vmd_loss, ts_pe_loss, ts_ce_loss, ts_le_loss, n_rad_loss]

    return mse






####
####
####
####                                                                                TRAINING LOOPS
####
####
####

def train_epoch(projectors, optimizers, mse_loss, train_loader, device, llm, beta):
    """
    Trains the projectors for one epoch using MSE and asymmetric loss functions.

    Parameters:
        projectors (list): List of projector models.
        optimizers (list): List of optimizers for each projector.
        mse_loss (function): Mean Squared Error loss function.
        train_loader (DataLoader): DataLoader for the training data.
        device (torch.device): Device to run the training on (CPU/GPU).
        llm (model): Language model for logit extraction.
        beta (float): Weighting factor for MSE loss.

    Returns:
        projectors (list): Updated list of trained projector models.
        train_loss_batches (list): List of training losses for each batch.
        gammas (list): List of gamma values for asymmetric loss.
        delta_ps (list): List of delta_p values for each class.

    Details:
        - Encodes and decodes inputs, computes embeddings.
        - Extracts logits using the language model.
        - Computes asymmetric and MSE losses.
        - Combines losses and performs backpropagation.
        - Clips gradients and updates optimizers.
    """

    sources = ['vd', 'vmd', 'ts_pe', 'ts_ce', 'ts_le', 'n_rad']
    gammas = [4,4,4,4,4,4,4,4,4,4,4,4]
    train_loss_batches = []

    for projector in projectors:
        projector.train()

    for batch_index, (x, y) in enumerate(train_loader, 1):

        for optimizer in optimizers:
            optimizer.zero_grad()

        asym_fns = [AsymmetricLoss(gamma_neg=gamma) for gamma in gammas]

        inputs, labels = x, y.to(device)


        encoded = []
        decoded = []
        mse_inputs = []
        for i,projector in enumerate(projectors):
            source_inputs = inputs[sources[i]].to(device)
            enc = projector.encoder(source_inputs)
            dec = projector.decoder(enc)
            mse_inputs.append(source_inputs)
            encoded.append(enc.view(-1,1,2048).to(torch.float16))
            decoded.append(dec)

        embeddings = torch.cat(encoded, dim=1).to(device)

        logits = logit_extractor(embeddings, llm)


        loss_asym, delta_ps = custom_asym_loss(logits, labels, asym_fns)
        loss_mse = custom_mse_loss(decoded, mse_inputs, mse_loss)

        losses = [loss_asym + beta * loss for loss in loss_mse]

        for i, optimizer in enumerate(optimizers):
            if i < len(optimizers) - 1:
               losses[i].backward(retain_graph=True)
            else:
               losses[i].backward()
            torch.nn.utils.clip_grad_norm_(projectors[i].parameters(), max_norm=1)
            optimizer.step()
            optimizer.zero_grad()


        loss = sum([loss.item() for loss in losses]) - 5*loss_asym.item()
        train_loss_batches.append(loss)

    return projectors, train_loss_batches, gammas, delta_ps




def validate(projectors, mse_loss, val_loader, device, llm, beta, gammas):
    """
    Validates the projectors using MSE and asymmetric loss functions.

    Parameters:
        projectors (list): List of projector models.
        mse_loss (function): Mean Squared Error loss function.
        val_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): Device to run the validation on (CPU/GPU).
        llm (model): Language model for logit extraction.
        beta (float): Weighting factor for MSE loss.
        gammas (list): List of gamma values for asymmetric loss.

    Returns:
        val_loss_batches (list): List of validation losses for each batch.
    """

    sources = ['vd', 'vmd', 'ts_pe', 'ts_ce', 'ts_le', 'n_rad']
    val_loss_batches = []

    for projector in projectors:
        projector.eval()
    
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):

            asym_fns = [AsymmetricLoss(gamma_neg=gamma) for gamma in gammas]

            inputs, labels = x, y.to(device)

            encoded = []
            decoded = []
            mse_inputs = []
            for i,projector in enumerate(projectors):
                source_inputs = inputs[sources[i]].to(device)
                enc = projector.encoder(source_inputs)
                dec = projector.decoder(enc)
                mse_inputs.append(source_inputs)
                encoded.append(enc.view(-1,1,2048).to(torch.float16))
                decoded.append(dec)

            embeddings = torch.cat(encoded, dim=1).to(device)

            logits = logit_extractor(embeddings, llm)


            loss_asym, delta_ps = custom_asym_loss(logits, labels, asym_fns)
            loss_mse = custom_mse_loss(decoded, mse_inputs, mse_loss)

            losses = [loss_asym + beta * loss for loss in loss_mse]

            loss = sum([loss.item() for loss in losses]) - 5*loss_asym.item()
            val_loss_batches.append(loss)

    return val_loss_batches




def training_loop(projectors, optimizers, mse_loss, train_loader, val_loader, num_epochs, llm, beta):
    """
    Runs the training loop for multiple epochs, saving model checkpoints and loss values.
    """

    print('Starting training')

    folder = 'test'
    train_losses, val_losses = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for projector in projectors:
        projector.to(device)

    
    for epoch in range(1,num_epochs+1):
        projectors, train_loss, gammas, delta_ps = train_epoch(projectors, optimizers, mse_loss, train_loader, device, llm, beta)

        val_loss = validate(projectors, mse_loss, val_loader, device, llm, beta, gammas)

        dp_values = [x.item() if isinstance(x, torch.Tensor) else x for x in delta_ps]

        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Val. loss: {sum(val_loss)/len(val_loss):.3f}, "
              f"Delta_p per class: {dp_values}")
            
        train_losses.append(sum(train_loss)/len(train_loss))
        val_losses.append(sum(val_loss)/len(val_loss))


        torch.save(projectors[0], f"{folder}/vd.pth")
        torch.save(projectors[1], f"{folder}/vmd.pth")
        torch.save(projectors[2], f"{folder}/ts_pe.pth")
        torch.save(projectors[3], f"{folder}/ts_ce.pth")
        torch.save(projectors[4], f"{folder}/ts_le.pth")
        torch.save(projectors[5], f"{folder}/n_rad.pth")

        with open(f"{folder}/train_losses.pkl", 'wb') as f1:
            pickle.dump(train_losses, f1)

        with open(f"{folder}/val_losses.pkl", 'wb') as f2:
            pickle.dump(val_losses, f2)
