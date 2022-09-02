import torch

def get_dist_pred_from_bnn(X_val, model, M):
    model.update_n_samples(n_samples=M)
    return model.forward_dist(X_val)[:,:,0]

def get_argmins_from_dist(sell_price, cost_price, dist):

    quantile_cut = (sell_price - cost_price)/sell_price
    argmin_from_dist = torch.quantile(
                        dist, 
                        quantile_cut, 
                        dim=0)
    return argmin_from_dist

def get_argmins_from_value(demand):
    argmin_from_value = demand
    return argmin_from_value

def profit_per_instance(order, demand_true, sell_price, cost_price):
    return sell_price*torch.minimum(order, demand_true) - cost_price*order

def profit_sum(order, demand_true, sell_price, cost_price):
    return profit_per_instance(order, demand_true, sell_price, cost_price).sum()

def compute_norm_regret_from_profits(profit_pred, profit_best):
    regret = profit_best - profit_pred
    norm_regret = regret/profit_best
    return norm_regret

def compute_norm_regret(X_val, y_val, model, M, sell_price, cost_price):
    if model.output_type_dist:
        Y_pred_dist = get_dist_pred_from_bnn(X_val, model, M)
        z_pred = get_argmins_from_dist(sell_price, cost_price, Y_pred_dist)

    else:
        Y_pred = model(X_val)
        z_pred = get_argmins_from_value(Y_pred[:,0])

    z_best = get_argmins_from_value(y_val[:,0])

    profit_pred = profit_sum(z_pred, y_val[:,0], sell_price, cost_price)
    profit_best = profit_sum(z_best, y_val[:,0], sell_price, cost_price)

    nr = compute_norm_regret_from_profits(profit_pred, profit_best)
    
    return nr


def compute_norm_regret_from_preds(X_val, y_val, Y_pred_dist, M, sell_price, cost_price):

    z_pred = get_argmins_from_dist(sell_price, cost_price, Y_pred_dist)
    z_best = get_argmins_from_value(y_val[:,0])

    profit_pred = profit_sum(z_pred, y_val[:,0], sell_price, cost_price)
    profit_best = profit_sum(z_best, y_val[:,0], sell_price, cost_price)

    nr = compute_norm_regret_from_profits(profit_pred, profit_best)
    
    return nr