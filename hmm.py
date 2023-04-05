def cal_output_probs(train_filename, output_filename):
    train_data = open(train_filename, 'r', encoding = "utf-8")
    output_probs = open(output_filename, 'w', encoding = "utf-8")
    dict_tags = {}
    dict_tokens = {}
    while True:
        line = train_data.readline().upper().split()
        if line == []:
            line2 = train_data.readline().upper().split()
            if line2 == []:
                break
            else:
                token = line2[0]
                tag  = line2[1]
                if tag not in dict_tags:
                    dict_tags[tag] = 1
                else:
                    dict_tags[tag] += 1
                if token not in dict_tokens:
                    dict_tokens[token] = {tag: 1}
                else:
                    if tag not in dict_tokens[token]:
                        dict_tokens[token][tag] = 1
                    else:
                        dict_tokens[token][tag] += 1
                continue

        token = line[0]
        tag = line[1]
        if tag not in dict_tags:
            dict_tags[tag] = 1
        else:
            dict_tags[tag] += 1
        if token not in dict_tokens:
            dict_tokens[token] = {tag: 1}
        else:
            if tag not in dict_tokens[token]:
                dict_tokens[token][tag] = 1
            else:
                dict_tokens[token][tag] += 1
    

    num_words = (len(dict_tokens.keys()))
    delta = 0.1
    dict_tokens["UNSEEN"] = {}

    for key, value in dict_tags.items():
        dict_tokens["UNSEEN"][key] = 0
        
##    print(dict_tokens["UNSEEN"])

    for key, value in dict_tokens.items():
        for x, y in value.items():
            joint_prob_tag_token = y
            num_of_counts_of_tag = dict_tags[x]
            cond_prob = (y + delta) / (dict_tags[x] + (num_words + 1)*delta)
            output_probs.write(f"{key} given {x} = {cond_prob:.8f}\n")
            ## print(f"P({key}|{x}) = {cond_prob:.8f}")
            

##cal_output_probs()

# Implement the six functions below

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    output_probs = open(in_output_probs_filename, 'r', encoding = "utf-8")
    test_data = open(in_test_filename, 'r', encoding = "utf-8")
    prediction = open(out_prediction_filename, 'w', encoding = "utf-8")
    dict_token_tag = {}

    while True:
        line = output_probs.readline().upper().split()
        if line == []:
            break
        token = line[0]
        tag = line[2]
        cond_prob = line[4]
        if token not in dict_token_tag:
            dict_token_tag[token] = (tag, float(cond_prob))
        else:
            curr_prob = dict_token_tag[token][1]
            if float(cond_prob) > curr_prob:
                dict_token_tag[token] = (tag, float(cond_prob))
            else:
                continue

    ##print(dict_token_tag)

    while True:
        line2 = test_data.readline().upper().split()
        if line2 ==[]:
            prediction.write('\n')
            line3 = test_data.readline().upper().split()
            if line3 == []:
                break
            else:
                given_token = line3[0]
                if given_token not in dict_token_tag:
                    best_tag = dict_token_tag["UNSEEN"][0]
                else:
                    best_tag = dict_token_tag[given_token][0]
                    
                prediction.write(best_tag + '\n')
            continue
                
        given_token = line2[0]
        if given_token not in dict_token_tag:
            best_tag = dict_token_tag["UNSEEN"][0]
        else:
            best_tag = dict_token_tag[given_token][0]
    
##        print((given_token, best_tag))
        prediction.write(best_tag + '\n')

##naive_predict('naive_output_probs.txt', 'twitter_dev_no_tag.txt', 'naive_predictions.txt')


def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    naive_output_probs = open(in_output_probs_filename, 'r', encoding="utf-8")
    train_data = open(in_train_filename, 'r', encoding="utf-8")
    test_data = open(in_test_filename, 'r', encoding="utf-8")
    prediction = open(out_prediction_filename, 'w', encoding="utf-8")

    ## get a dictionary of w:count(x=w) & a dictionary of j:count(y=j)
    word_dictionary = {}
    tag_dictionary = {}
    
    while True:
        line = train_data.readline().split() #[token, tag]
        if line != []:
            word = line[0]
            tag = line[1]

        elif line == []:
            line2 = train_data.readline().split() #[user, @]
            if line2 == []:
                break
            else:
                word = line2[0]
                tag = line2[1]

        if word not in word_dictionary:
            word_dictionary[word] = 1
        else:
            word_dictionary[word] += 1

        if tag not in tag_dictionary:
            tag_dictionary[tag] = 1
        else:
            tag_dictionary[tag] += 1

    
    ## GIVEN WORD, BEST TAG!!
    ## get P(y=j|x=w)
    final_dict = {} ## {word: [best tag, prob],...}
    while True:
        line = naive_output_probs.readline().split() #[word, tag, probability]
        if line == []:
            break;
        else:
            word = line[0]
            tag = line[2]
            prob = float(line[4])
            q3_prob = prob * (tag_dictionary[tag]/sum(tag_dictionary.values()))

            if word not in final_dict:
                final_dict[word] = [tag, q3_prob]
            else:
                if q3_prob > final_dict[word][1]:
                    final_dict[word] = [tag, q3_prob]

    while True:
        line2 = test_data.readline().upper().split()
        if line2 ==[]:
            prediction.write('\n')
            line3 = test_data.readline().upper().split()
            if line3 == []:
                break
            else:
                given_token = line3[0]
                if given_token not in final_dict:
                    best_tag = final_dict["UNSEEN"][0]
                else:
                    best_tag = final_dict[given_token][0]
                    
                prediction.write(best_tag + '\n')
            continue
                
        given_token = line2[0]
        if given_token not in final_dict:
            best_tag = final_dict["UNSEEN"][0]
        else:
            best_tag = final_dict[given_token][0]
    
##        print((given_token, best_tag))
        prediction.write(best_tag + '\n')
        
## naive_predict2("naive_output_probs.txt", "twitter_train.txt", "twitter_dev_no_tag.txt", "naive_predictions2.txt")



def viterbi_a(train_filename, output_filename, trans_filename):
    train_data = open(train_filename, 'r', encoding="utf-8")
    #output_probs = open(output_filename, 'w', encoding="utf-8")
    trans_probs = open(trans_filename, 'w', encoding="utf-8")

    ## compute the output probabilities
    cal_output_probs(train_filename, output_filename)

    ## compute the transition probabilities
    ## make a tag_dict -- {tag i : {tag i+1 : count}}
    ## numerator = count, denominator = sum all values in tag i

    tag_dict = {}
    
    tag_j = "START" ## first tag
    
    while True:
        line = train_data.readline().split()
        if line != []:
            tag_i = tag_j
            tag_j = line[1]

            if tag_i not in tag_dict:
                tag_dict[tag_i] = {tag_j: 1}
            else:
                if tag_j not in tag_dict[tag_i].keys():
                    tag_dict[tag_i][tag_j] = 1
                else:
                    tag_dict[tag_i][tag_j] += 1

            

        elif line == []:
            ## not showing ..??
            
            if tag_j not in tag_dict:
                tag_dict[tag_j] = {"STOP": 1}
                
            else:
                if "STOP" not in tag_dict[tag_j].keys():
                    tag_dict[tag_i]["STOP"] = 1
                else:
                    tag_dict[tag_j]["STOP"] += 1

                line2 = train_data.readline().split()
                if line2 == []:
                    break
            
                else:
                    tag_i = "START"
                    tag_j = line2[1]

                    if "START" not in tag_dict:
                        tag_dict["START"] = {"START": 1}
                    else:
                        if tag_j not in tag_dict["START"].keys():
                            tag_dict["START"][tag_j] = 1
                        else:
                            tag_dict["START"][tag_j] += 1


    ## calc each (tag_j|tag_i) w sus smoothing
    delta = 0.1
    
    for currTag, tag_j_dict in tag_dict.items():
        for nextTag, count in tag_j_dict.items():
            numerator = count
            denominator = sum(tag_j_dict.values())
            num_words = len(tag_dict.keys())
            transition_prob = (numerator + delta)/(denominator + delta*(num_words+1))
            trans_probs.write(f"{currTag} to {nextTag} = {transition_prob:.8f}\n")


    
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    
    pass

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass




def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

##print(evaluate('naive_predictions.txt', 'twitter_dev_ans.txt'))

def run():

    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''


    ddir = '' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


##if __name__ == '__main__':
##    run()


