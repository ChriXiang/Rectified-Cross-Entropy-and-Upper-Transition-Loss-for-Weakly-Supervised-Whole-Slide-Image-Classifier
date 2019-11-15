def RCE_UT_loss(y_true, y_pred, weights):
    weights = K.reshape(weights, (K.shape(weights)[0], 1))
    
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    #print(y_pred.get_shape())
    loss = y_true * K.log(y_pred) * weights
    rce_loss = -K.sum(loss, -1)
    #====
    l,c = y_pred.get_shape()
    preds = tf.split(y_pred,c,axis=1)
    #print(y_true.get_shape(),y_pred.get_shape())
    l,c = y_true.get_shape()
    truth = tf.split(y_true,2,axis=1)

    loss2 = Lambda(lambda a: K.switch(K.greater(a[-1],truth[-1]),a[-1],K.zeros_like(a[-1])))(preds)

    ones = K.ones_like(loss2)
    tensor = Subtract()([ones,loss2])
    loss2 = K.log(tensor)

    up_loss = -K.sum(loss2,-1)

    loss = rce_loss + 10*up_loss

    return loss
