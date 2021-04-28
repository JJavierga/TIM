import numpy as np
from sacred import Ingredient
from src.utils import warp_tqdm, compute_confidence_interval, load_checkpoint
from src.utils import load_pickle, save_pickle
from src.datasets.ingredient import get_dataloader
import os
import torch
import collections
import torch.nn.functional as F
from src.timV2 import TIM, TIM_ADM, TIM_GD

import csv

eval_ingredient = Ingredient('eval')
@eval_ingredient.config
def config():
    number_tasks = 1
    n_ways = 5
    query_shots = 15
    method = 'baseline'
    model_tag = 'best'
    target_data_path = None  # Only for cross-domain scenario
    target_split_dir = None  # Only for cross-domain scenario
    plt_metrics = ['accs']
    shots = 10 #[1, 5]
    used_set = 'test'  # can also be val for hyperparameter tuning
    fresh_start = True
    checking = True


class Evaluator:
    @eval_ingredient.capture
    def __init__(self, device, ex):
        self.device = device
        self.ex = ex

    @eval_ingredient.capture
    def run_full_evaluation(self, model, model_path, model_tag, shots, method, callback, target_split_dir, checking):
        """
        Run the evaluation over all the tasks in parallel
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            method : Which method to use for inference ("baseline", "tim-gd" or "tim-adm")
            shots : Number of support shots to try

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        print("=> Runnning full evaluation with method: {}".format(method))

        # Load pre-trained model
        load_checkpoint(model=model, model_path=model_path, type=model_tag)

        # Get loaders
        loaders_dic = self.get_loaders()

        # Extract features (just load them if already in memory)
        extracted_features_shots_dic = self.extract_features_shots(model=model,
                                                       model_path=model_path,
                                                       loaders_dic=loaders_dic)

        extracted_features_queries_dic = self.extract_features_queries(model=model,
                                                        model_path=model_path,
                                                        loaders_dic=loaders_dic)

        results = []
    
        tasks = self.generate_task(extracted_features_shots_dic=extracted_features_shots_dic,
                                    extracted_features_queries_dic=extracted_features_queries_dic, 
                                    shot=shots)

        logs_prob, logs_y = self.run_task(task_dic=tasks,
                                model=model,
                                callback=callback)
    
        classes =  {'nodefect':0,'scaling':1, 'efflorescence':2, 'cracks':3, 'spalling':4}
        right_elems = {'nodefect':0, 'scaling':0, 'efflorescence':0, 'cracks':0, 'spalling':0}
        total_elems = {'nodefect':0, 'scaling':0, 'efflorescence':0, 'cracks':0, 'spalling':0}
        
        #classes =  {'nodefect':0, 'defect':1}
        #total_elems = {'nodefect':0, 'defect':1}
        #right_elems = {'nodefect':0,  'defect':1}
        inv_map = {v: k for k, v in classes.items()}

        with open(os.path.join(target_split_dir, "query.csv")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            # Checking results:
            if checking == True:                                                   
                total_positives = 0
                true_positives = 0
                false_positives = 0
                true_negatives = 0
                for row in csv_reader:
                    if line_count == 0:
                        pass
                    else:
                        if classes[row[1]]==logs_y[0][line_count-1]:
                            right_elems[row[1]]+=1                           
                        else:
                            print("Confused {} with {} \t Probabilities: {} \t Image: {}".format(row[1], inv_map[logs_y[0][line_count-1]], logs_prob[0][line_count-1], row[0]))
                        total_elems[row[1]]+=1
                        if logs_y[0][line_count-1]!=0:
                            total_positives += 1
                            if classes[row[1]]!=0:
                                true_positives += 1
                            else:
                                false_positives += 1           
                        else:
                            if classes[row[1]]==0:
                                true_negatives += 1
                            else:
                                false_negatives += 1
                    line_count += 1
                print(f'Processed {line_count-1} images.')   
                global_right=0    
                for class_i in classes:
                    print('Accuracy for {}: {:.4f}'.format(class_i, right_elems[class_i]/total_elems[class_i]))
                    global_right+=right_elems[class_i]
                print('Global accuracy: {:.4f}'.format(global_right/(line_count-1)))
                print('Positive = defect: \n\tFalse positives: {}\n\tFalse negatives: {}\n\tTrue positives: {}\n\tTrue negatives: {}'.format(false_positives, total_positives-true_positives, true_positives, true_negatives))

            else:
                for row in csv_reader:
                    if line_count == 0:
                        pass
                    else:
                        print("Image: {} \t is {} \t with probabilities: {}".format(row[0], inv_map[logs_y[0][line_count-1]], logs_prob[0][line_count-1] ))
                        total_elems[inv_map[logs_y[0][line_count-1]]] += 1
                    line_count += 1
                print("\n")
                for class_i in classes:
                    print('Images of class {}: \t{:.0f}'.format(class_i, total_elems[class_i]))

        return results

    def run_task(self, task_dic, model, callback):

        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model=model)

        # Extract support and query
        y_s = task_dic['y_s']
        z_s, z_q = task_dic['z_s'], task_dic['z_q']

        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)


        # Perform normalizations required
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)

        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s)
        tim_builder.init_weights(support=support, y_s=y_s, query=query)

        # Run adaptation
        probs, results = tim_builder.run_adaptation(support=support, query=query, y_s=y_s, callback=callback)

        return probs, results

    @eval_ingredient.capture
    def get_tim_builder(self, model, method):
        # Initialize TIM classifier builder
        tim_info = {'model': model}
        if method == 'tim_adm':
            tim_builder = TIM_ADM(**tim_info)
        elif method == 'tim_gd':
            tim_builder = TIM_GD(**tim_info)
        elif method == 'baseline':
            tim_builder = TIM(**tim_info)
        else:
            raise ValueError("Method must be in ['tim_gd', 'tim_adm', 'baseline']")
        return tim_builder

    @eval_ingredient.capture
    def get_loaders(self, used_set, target_data_path, target_split_dir):
        # First, get loaders
        loaders_dic = {}
        loader_info = {'aug': False, 'shuffle': False, 'out_name': False}

        if target_data_path is not None:  # This mean we are in the cross-domain scenario
            loader_info.update({'path': target_data_path,
                                'split_dir': target_split_dir})

        train_loader = get_dataloader('support', **loader_info)
        loaders_dic['support'] = train_loader

        test_loader = get_dataloader('query', **loader_info)
        loaders_dic.update({'query': test_loader})
        return loaders_dic

    @eval_ingredient.capture
    def extract_features_shots(self, model, model_path, model_tag, used_set, fresh_start, loaders_dic):
        """
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            used_set : Set used between 'test' and 'val'
            n_ways : Number of ways for the task

        returns :
            extracted_features_dic : Dictionnary containing all extracted features and labels
        """

        # Load features from memory if previously saved ...
        save_dir = os.path.join(model_path, model_tag, used_set)
        filepath = os.path.join(save_dir, 'output_support.plk')
        if os.path.isfile(filepath) and (not fresh_start):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic

        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            os.makedirs(save_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():

            all_features = []
            all_labels = []
            for i, (inputs, labels, _) in enumerate(warp_tqdm(loaders_dic['support'], False)):
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs, True)
                all_features.append(outputs.cpu())
                all_labels.append(labels)
            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            extracted_features_dic = {'concat_features': all_features,
                                      'concat_labels': all_labels
                                      }
        print(" ==> Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic


    @eval_ingredient.capture
    def extract_features_queries(self, model, model_path, model_tag, used_set, fresh_start, loaders_dic):
        """
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            used_set : Set used between 'test' and 'val'
            n_ways : Number of ways for the task

        returns :
            extracted_features_dic : Dictionnary containing all extracted features and labels
        """

        # Load features from memory if previously saved ...
        save_dir = os.path.join(model_path, model_tag, used_set)
        filepath = os.path.join(save_dir, 'output_query.plk')
        if os.path.isfile(filepath) and (not fresh_start):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic

        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            os.makedirs(save_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():

            all_features = []
            all_labels = []
            for i, (inputs, _, _) in enumerate(warp_tqdm(loaders_dic['query'], False)):
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs, True)
                all_features.append(outputs.cpu())
            all_features = torch.cat(all_features, 0)
            extracted_features_dic = {'concat_features': all_features}
        print(" ==> Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic


    @eval_ingredient.capture
    def get_task(self, shot, n_ways, query_shots, extracted_features_shots_dic, extracted_features_queries_dic):
        """
        inputs:
            extracted_features_dic : Dictionnary containing all extracted features and labels
            shot : Number of support shot per class
            n_ways : Number of ways for the task

        returns :
            task : Dictionnary : z_support : torch.tensor of shape [n_ways * shot, feature_dim]
                                 z_query : torch.tensor of shape [n_ways * query_shot, feature_dim]
                                 y_support : torch.tensor of shape [n_ways * shot]
                                 y_query : torch.tensor of shape [n_ways * query_shot]
        """
        shots_features = extracted_features_shots_dic['concat_features']
        queries_features =  extracted_features_queries_dic['concat_features']
    
        all_labels = extracted_features_shots_dic['concat_labels']
        all_classes = torch.unique(all_labels)
        samples_classes = np.random.choice(a=all_classes, size=n_ways, replace=False)
        support_samples = []
        query_samples = []
        
        support_samples.append(shots_features)
        query_samples.append(queries_features)

        y_support = torch.arange(n_ways)[:, None].repeat(1, shot).reshape(-1)
        #print(y_support)
        z_support = torch.cat(support_samples, 0)
        z_query = torch.cat(query_samples, 0)

        task = {'z_s': z_support, 'y_s': y_support,
                'z_q': z_query}
        return task

    @eval_ingredient.capture
    def generate_task(self, extracted_features_queries_dic, shot, number_tasks, extracted_features_shots_dic):
        """
        inputs:
            extracted_features_dic :
            shot : Number of support shot per class
            number_tasks : Number of tasks to generate

        returns :
            merged_task : { z_support : torch.tensor of shape [number_tasks, n_ways * shot, feature_dim]
                            z_query : torch.tensor of shape [number_tasks, n_ways * query_shot, feature_dim]
                            y_support : torch.tensor of shape [number_tasks, n_ways * shot]
                            y_query : torch.tensor of shape [number_tasks, n_ways * query_shot] }
        """
        print(f" ==> Generating 1 task ...")
        task_dic = self.get_task(shot=shot, extracted_features_queries_dic=extracted_features_queries_dic, extracted_features_shots_dic=extracted_features_shots_dic)

        # Now merging all tasks into 1 single dictionnary
        merged_tasks = {}
        n_tasks = len(task_dic)
        for key in task_dic.keys():
            n_samples = task_dic[key].size(0)
            merged_tasks[key] = torch.cat([task_dic[key] for i in range(1)], dim=0).view(1, n_samples, -1)
        return merged_tasks
