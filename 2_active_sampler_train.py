import random

def weighted_sampling(samples, weights):
    """
    根据样本的权重进行加权采样
    :param samples: 样本列表
    :param weights: 对应样本的权重列表
    :return: 加权采样的样本列表
    """
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    return random.choices(samples, probabilities, k=len(samples))

def build_sampler_graph(n_nodes, edge_threshold, graph):
    adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)
    edge_matrix = torch.zeros(n_nodes, edge_threshold)

    """sample neighbors for each node"""
    for node in tqdm(graph.nodes, ascii=True, desc="Build active-sampler matrix"):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= edge_threshold:
            # 假设我们有一个置信度列表，这里简单地使用随机生成的置信度
            confidence_scores = [random.random() for _ in range(len(neighbors))]
            sampled_edge = weighted_sampling(neighbors, confidence_scores)
            edges = deepcopy(sampled_edge)
        else:
            neg_id = random.sample(
                range(global_kg.item_range[0], global_kg.item_range[1] + 1),
                edge_threshold - len(neighbors),
            )
            node_id = [node] * (edge_threshold - len(neighbors))
            sampled_edge = neighbors + neg_id
            edges = neighbors + node_id

        adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
        edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.cuda().long()
        edge_matrix = edge_matrix.cuda().long()

    return adj_matrix, edge_matrix

def train(graph, args_config):
    """build padded training set"""
    train_mat = graph.train_user_dict
    train_data = build_train_data(train_mat)

    sampler = ALPolicy(args_config)
    ####for break point
    start_epoch = args_config.pretrain_as_epoch
    if args_config.pretrain_as_epoch:  # 不为零
        print("\nactive-sampler break in epoch {}, recover now!!!!!".format(start_epoch))
        model_dict = load_as_model(epoch=start_epoch)
        sampler.load_state_dict(model_dict)
    cuda_(sampler)
    print("\nSet active-sampler as: {}".format(str(sampler)))

    sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr=args_config.sllr)

    for epoch in range(start_epoch, args_config.epoch):
        print('##############################################################')
        print("Epoch {}/{} ".format(epoch, args_config.epoch))
        u, item = bcfg.train_list[epoch]
        user_id = int(u)
        item_id = int(item)
        preference_list = bcfg.item_dict[str(item_id)]['feature_index']

        the_agent = ALagent(user_id, item_id, preference_list, sampler, sampler_optimizer, graph)

        """Train one epoch"""
        shapedrewards, logp_actions, p_actions = the_agent.playOneEpisode(epoch)
        loss = the_agent.finishEpisode(shapedrewards, logp_actions, p_actions)

        print("\nActive Sampler Agent loss : {}\n".format(loss))

        sampler = deepcopy(the_agent.policy)
        if epoch % 1000 == 0 and epoch != args_config.epoch - 1:
            save_active_sampler_model(the_agent.policy, epoch)

        """VALIDATE"""
        if epoch % args_config.show_step == 0:
            with torch.no_grad():
                score, bia = mean_std(the_agent.cur_rewards)
            print('-----------epoch {} evaluate result--------------'.format(epoch))
            print('score : {} , bia : {} '.format(score, bia))
            print('-------------------------------------------------')

        """FOR TEST"""
        if epoch == args_config.epoch - 1:
            the_evaluate_agent = ALagent(user_id, item_id, preference_list, sampler, sampler_optimizer, graph, test=True)

    save_final_active_sampler_model(sampler)

    """TEST"""
    the_evaluate_agent.test = True
    finalrewards = the_evaluate_agent.get_reward(args_config.mt)
    # print('finalrewards:{}'.format(finalrewards))
    score, bia = mean_std(finalrewards)
    print('---------------------FINAL test result---------------------')
    print('score : {} , bia : {} '.format(score, bia))
    print('-----------------------------------------------------------')


if __name__ == "__main__":
    """initialize dataset"""
    # global_kg = global_kg

    """initialize args"""
    A = bcfg.get_Active_Sampler_parser()

    data_config = {
        "n_users": global_kg.n_users,
        "n_items": global_kg.n_items,
        "n_relations": global_kg.n_relations + 2,
        "n_entities": global_kg.n_entities,
        "n_nodes": global_kg.entity_range[1] + 1,
        "item_range": global_kg.item_range,
    }

    # """fix the random seed"""
    # A.seed = 2021
    # random.seed(A.seed)
    # np.random.seed(A.seed)
    # torch.manual_seed(A.seed)
    # torch.cuda.manual_seed(A.seed)


    train(
        graph=global_kg,
        args_config=A
    )
