#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class DHBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]
        self.alpha1 = self.conf['alpha_1']
        self.alpha2 = self.conf['alpha_2']

        self.hyper_num = self.conf['hyper_num']
        self.h_m = self.conf['h_m']
        self.hyper_ratio = self.conf['hyper_ratio']
        self.act = nn.LeakyReLU(negative_slope=0.05)
        
        self.UI_mlp_users = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])
        self.UI_mlp_items = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])
        self.UB_mlp_bundles = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])
        self.UB_mlp_users = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])
        self.BI_mlp_bundles = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])
        self.BI_mlp_items = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_size, self.hyper_num), nn.ReLU(), nn.Linear(self.hyper_num, self.hyper_num, bias=False), nn.Softmax(dim=1)) for i in range(self.num_layers)])

        self.hyper_weight1 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.hyper_weight2 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))
        self.hyper_weight3 = nn.Parameter(torch.nn.init.xavier_normal_(torch.FloatTensor(self.hyper_num, self.hyper_num)))

        self.fusion_weights = conf['fusion_weights']

        self.init_emb()
        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        # generate the graph without any dropouts for testing
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)

        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)

        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        if self.conf['aug_type'] == 'MD':
            self.init_md_dropouts()
        elif self.conf['aug_type'] == "Noise":
            self.init_noise_eps()

    def hyper_Propagate(self, feature, adj, weight1_hyper, weight2_hyper, weight3_hyper):
        feature1 = self.act(adj.T @ feature)
        if self.h_m == 0 :
            return self.act(adj @ feature1)
        feature2 = self.act(weight1_hyper @ feature1) + feature1
        if self.h_m == 1 :
            feature = self.act(adj @ feature2)
            return feature
        feature3 = self.act(weight2_hyper @ feature2) + feature2
        if self.h_m == 2 :
            feature = self.act(adj @ feature3)
        else:    
            feature4 = self.act(weight3_hyper @ feature3) + feature3
            feature = self.act(adj @ feature4)
        return feature
    
    def init_md_dropouts(self):
        self.UB_dropout = nn.Dropout(self.conf["UB_ratio"], True)
        self.UI_dropout = nn.Dropout(self.conf["UI_ratio"], True)
        self.BI_dropout = nn.Dropout(self.conf["BI_ratio"], True)
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout
        }


    def init_noise_eps(self):
        self.UB_eps = self.conf["UB_ratio"]
        self.UI_eps = self.conf["UI_ratio"]
        self.BI_eps = self.conf["BI_ratio"]
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def init_fusion_weights(self):
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"

        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1),\
            "The number of layer fusion weights does not correspond to number of layers"

        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)


    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(device)


    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bipartite_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)


    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
        all_hyper_features = [features]
        all_fixed_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features) 
            
            if self.conf["aug_type"] == "MD" and not test:
                mess_dropout = self.mess_dropout_dict[graph_type]
                features = mess_dropout(features)
            elif self.conf["aug_type"] == "Noise" and not test:
                random_noise = torch.rand_like(features).to(self.device)
                eps = self.eps_dict[graph_type]
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps
            A_feature, B_feature = torch.split(features, (A_feature.shape[0], B_feature.shape[0]), 0)

            if graph_type == 'UB':
                A_hypergraph = self.UB_mlp_users[i](A_feature)
                B_hypergraph = self.UB_mlp_bundles[i](B_feature)
                # A_hypergraph = nn.Parameter(torch.FloatTensor(self.num_users, self.hyper_num).unsqueeze(0)).to(self.device)
                # nn.init.xavier_normal_(A_hypergraph)
                # B_hypergraph = nn.Parameter(torch.FloatTensor(self.num_bundles, self.hyper_num).unsqueeze(0)).to(self.device)
                # nn.init.xavier_normal_(B_hypergraph)
            elif graph_type == 'UI':
                A_hypergraph = self.UI_mlp_users[i](A_feature)
                B_hypergraph = self.UI_mlp_items[i](B_feature)
                # A_hypergraph = nn.Parameter(torch.FloatTensor(self.num_users, self.hyper_num).unsqueeze(0)).to(self.device)
                # nn.init.xavier_normal_(A_hypergraph)
                # B_hypergraph = nn.Parameter(torch.FloatTensor(self.num_items, self.hyper_num).unsqueeze(0)).to(self.device)
                # nn.init.xavier_normal_(B_hypergraph)
            elif graph_type == 'BI':
                A_hypergraph = self.BI_mlp_bundles[i](A_feature)
                B_hypergraph = self.BI_mlp_items[i](B_feature)
                # A_hypergraph = nn.Parameter(torch.FloatTensor(self.num_bundles, self.hyper_num).unsqueeze(0)).to(self.device)
                # nn.init.xavier_normal_(A_hypergraph)
                # B_hypergraph = nn.Parameter(torch.FloatTensor(self.num_items, self.hyper_num).unsqueeze(0)).to(self.device)
                # nn.init.xavier_normal_(B_hypergraph)

            A_hypergraph = A_hypergraph.squeeze(0)
            B_hypergraph = B_hypergraph.squeeze(0)

            A_hypergraph = F.dropout(A_hypergraph, p = 1 - self.hyper_ratio)
            B_hypergraph = F.dropout(B_hypergraph, p = 1 - self.hyper_ratio)

            hyper_A_feature = self.hyper_Propagate(A_feature, A_hypergraph, self.hyper_weight1, self.hyper_weight2, self.hyper_weight3)
            hyper_B_feature = self.hyper_Propagate(B_feature, B_hypergraph, self.hyper_weight1, self.hyper_weight2, self.hyper_weight3)
            
            hyper_features = torch.cat((hyper_A_feature, hyper_B_feature), 0)

            # if self.conf["aug_type"] == "MD" and not test:
            #     mess_dropout = self.mess_dropout_dict[graph_type]
            #     hyper_features = mess_dropout(hyper_features)
            # elif self.conf["aug_type"] == "Noise" and not test:
            #     random_noise = torch.rand_like(hyper_features).to(self.device)
            #     eps = self.eps_dict[graph_type]
            #     hyper_features += torch.sign(hyper_features) * F.normalize(random_noise, dim=-1) * eps
            
            fixed_features = (self.alpha1 * features + self.alpha2 * hyper_features)

            # if self.conf["aug_type"] == "MD" and not test:
            #     mess_dropout = self.mess_dropout_dict[graph_type]
            #     fixed_features = mess_dropout(fixed_features)
            # elif self.conf["aug_type"] == "Noise" and not test:
            #     random_noise = torch.rand_like(fixed_features).to(self.device)
            #     eps = self.eps_dict[graph_type]
            #     fixed_features += torch.sign(fixed_features) * F.normalize(random_noise, dim=-1) * eps


            all_features.append(F.normalize(features, p=2, dim=1))
            all_hyper_features.append(F.normalize(hyper_features, p=2, dim=1))
            all_fixed_features.append(F.normalize(fixed_features, p=2, dim=1))


        
        all_features = torch.stack(all_features, 1) * layer_coef
        all_features = torch.sum(all_features, dim=1)

        all_hyper_features = torch.stack(all_hyper_features, 1) * layer_coef
        all_hyper_features = torch.sum(all_hyper_features, dim=1)

        all_fixed_features = torch.stack(all_fixed_features, 1) * layer_coef
        all_fixed_features = torch.sum(all_fixed_features, dim=1).squeeze(1)


        A_ori_feature, B_ori_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        A_hyper_feature, B_hyper_feature = torch.split(all_hyper_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        A_fixed_feature, B_fixed_feature = torch.split(all_fixed_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_ori_feature, B_ori_feature, A_hyper_feature, B_hyper_feature, A_fixed_feature, B_fixed_feature


    def aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["aug_type"] == "MD" and not test:
            mess_dropout = self.mess_dropout_dict[graph_type]
            aggregated_feature = mess_dropout(aggregated_feature)
        elif self.conf["aug_type"] == "Noise" and not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            eps = self.eps_dict[graph_type]
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature


    def fuse_users_bundles_feature(self, users_feature, bundles_feature):
        users_feature = torch.stack(users_feature, dim=0)
        bundles_feature = torch.stack(bundles_feature, dim=0)

        # Modal aggregation
        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        # users_rep = torch.sum(users_feature, dim=0)
        # bundles_rep = torch.sum(bundles_feature, dim=0)

        return users_rep, bundles_rep


    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        if test:
            UB_users_ori_feature, UB_bundles_ori_feature, UB_users_hyper_feature, UB_bundles_hyper_feature, UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            UB_users_ori_feature, UB_bundles_ori_feature, UB_users_hyper_feature, UB_bundles_hyper_feature, UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        if test:
            UI_users_ori_feature, UI_items_ori_feature, UI_users_hyper_feature, UI_items_hyper_feature, UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
        else:
            UI_users_ori_feature, UI_items_ori_feature, UI_users_hyper_feature, UI_items_hyper_feature, UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)

        #  =============================  BI graph propagation  =============================
        if test:
            BI_bundles_ori_feature, BI_items_ori_feature, BI_bundles_hyper_feature, BI_items_hyper_feature, BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
        else:
            BI_bundles_ori_feature, BI_items_ori_feature, BI_bundles_hyper_feature, BI_items_hyper_feature, BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)

        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]

        users_rep, bundles_rep = self.fuse_users_bundles_feature(users_feature, bundles_feature)

        return users_rep, bundles_rep, UB_users_feature, UI_users_feature, UB_bundles_feature, UI_bundles_feature


    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss
    
    def cal_diff_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :] 
        aug = aug[:, 0, :] 

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        #pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        #pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(1 / ttl_score))

        return c_loss
    
    def cal_allign_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :] 
        aug = aug[:, 0, :] 

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        #ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        #ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score))

        return c_loss


    def cal_loss(self, users_feature, bundles_feature):
        # users_feature / bundles_feature: [bs, 1+neg_num, emb_size]
        pred = torch.sum(users_feature * bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        # cl is abbr. of "contrastive loss"
        u_view_cl = self.cal_c_loss(users_feature, users_feature)
        b_view_cl = self.cal_c_loss(bundles_feature, bundles_feature)

        c_losses = [u_view_cl, b_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss


    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_rep, bundles_rep, UB_users_feature, UI_users_feature, UB_bundles_feature, UI_bundles_feature = self.get_multi_modal_representations()

        # ori_users_feature = [i[users].expand(-1, bundles.shape[1], -1) for i in ori_users_feature]
        # UI_ori_users_feature, UB_ori_users_feature = ori_users_feature

        # ori_bundles_feature = [i[bundles] for i in ori_bundles_feature]
        # UB_ori_bundles_feature, BI_ori_bundles_feature = ori_bundles_feature

        # hyper_users_feature = [i[users].expand(-1, bundles.shape[1], -1) for i in hyper_users_feature]
        # UI_hyper_users_feature, UB_hyper_users_feature = hyper_users_feature

        # hyper_bundles_feature = [i[bundles] for i in hyper_bundles_feature]
        # UB_hyper_bundles_feature, BI_hyper_bundles_feature = hyper_bundles_feature
        # ori_users_embedding = ori_users_rep[users].expand(-1, bundles.shape[1], -1)
        # ori_bundles_embedding = ori_bundles_rep[bundles]

        # hyper_users_embedding = hyper_users_rep[users].expand(-1, bundles.shape[1], -1)
        # hyper_bundles_embedding = hyper_bundles_rep[bundles]

        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]

        UB_users_embedding = UB_users_feature[users].expand(-1, bundles.shape[1], -1)
        UB_bundles_embedding = UB_bundles_feature[bundles]

        UI_users_embedding = UI_users_feature[users].expand(-1, bundles.shape[1], -1)
        UI_bundles_embedding = UI_bundles_feature[bundles]
        
        uu_c_loss = self.cal_c_loss(UB_users_embedding, UI_users_embedding)
        bb_c_loss = self.cal_c_loss(UB_bundles_embedding, UI_bundles_embedding)
        # users_c_loss = self.cal_c_loss(ori_users_embedding, hyper_users_embedding)
        # bundles_c_loss = self.cal_c_loss(ori_bundles_embedding, hyper_bundles_embedding)
        ub_c_losses = [uu_c_loss, bb_c_loss]
        ub_c_loss = sum(ub_c_losses) / len(ub_c_losses)
        # oh_c_losses = [users_c_loss + bundles_c_loss]
        # oh_c_loss = sum(oh_c_losses) / len(oh_c_losses)

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss, ub_c_loss


    def evaluate(self, users_feature, bundles_feature,  users):
        # users_feature, bundles_feature = propagate_result
        scores = torch.mm(users_feature[users], bundles_feature.t())
        return scores
