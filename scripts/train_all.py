import train_catboost,train_forest,train_linear

if __name__=='__main__':
    train_linear.start()
    train_forest.start()
    train_catboost.start()