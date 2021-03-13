def get_params():
    # for 'ogbg-molhiv'
    num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 5, [32, 1], 64, 16, 16
    return num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim
