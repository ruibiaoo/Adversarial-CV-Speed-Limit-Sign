import tensorflow as tf

def boosted_cross_entropy(y_true, y_pred):
    # Clip probabilities for numerical stability
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    
    # 1. Standard Cross-Entropy: -sum(y_true * log(y_pred))
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    
    # 2. Mask out the true class to find the highest-probability wrong class
    wrong_class_probs = y_pred * (1.0 - y_true)
    max_wrong_prob = tf.reduce_max(wrong_class_probs, axis=-1)
    
    # 3. Runner-up suppression penalty: -log(1 - max_{k != y} p_k)
    margin_loss = -tf.math.log(1.0 - max_wrong_prob + eps)
    
    return tf.reduce_mean(ce_loss + margin_loss)