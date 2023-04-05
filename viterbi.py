def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    :param obs: a sequence of observations
    :param states: a set of possible hidden states
    :param start_p: a dictionary of starting probabilities for each state
    :param trans_p: a dictionary of transition probabilities between states
    :param emit_p: a dictionary of emission probabilities for each state and observation pair
    :return: a tuple of (best hidden state sequence, probability of that sequence)
    """
    V = [{}]
    path = {}

    # Initialize base cases (t=0)
    for state in states:
        V[0][state] = start_p[state] * emit_p[state][obs[0]]
        path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for state in states:
            (prob, prev_state) = max((V[t-1][prev_state] * trans_p[prev_state][state] * emit_p[state][obs[t]], prev_state) for prev_state in states)
            V[t][state] = prob
            new_path[state] = path[prev_state] + [state]

        path = new_path

    # Find the probability and path of the best sequence
    (prob, state) = max((V[len(obs)-1][state], state) for state in states)
    return (path[state], prob)
