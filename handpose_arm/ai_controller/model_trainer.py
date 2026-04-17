import os
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        self.best_model = None
        self.best_score = -float('inf')
        
    def load_data(self, data_file="data/demonstrations.json"):
        if not os.path.exists(data_file):
            print(f"[错误] 数据文件不存在: {data_file}")
            return None, None
        
        with open(data_file, 'r', encoding='utf-8') as f:
            demonstrations = json.load(f)
        
        X = []
        y = []
        
        for demo in demonstrations:
            for sample in demo["samples"]:
                X.append(sample["keypoints"])
                y.append(sample["servos"])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"[数据加载] X: {X.shape}, y: {y.shape}")
        return X, y
    
    def normalize_data(self, X, y):
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X = X_scaler.fit_transform(X)
        y = y_scaler.fit_transform(y)
        
        self.scalers['X'] = X_scaler
        self.scalers['y'] = y_scaler
        
        return X, y
    
    def train_knn(self, X_train, y_train, X_val, y_val, n_neighbors=5):
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"  KNN: train R2={train_score:.4f}, val R2={val_score:.4f}")
        return model, val_score
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, n_estimators=100, max_depth=10):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"  RF: train R2={train_score:.4f}, val R2={val_score:.4f}")
        return model, val_score
    
    def train_mlp(self, X_train, y_train, X_val, y_val, hidden_sizes=(128, 64, 32)):
        model = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            activation='relu',
            solver='adam',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"  MLP: train R2={train_score:.4f}, val R2={val_score:.4f}")
        return model, val_score
    
    def train_gboost(self, X_train, y_train, X_val, y_val, n_estimators=100):
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"  GBoost: train R2={train_score:.4f}, val R2={val_score:.4f}")
        return model, val_score
    
    def evaluate(self, model, X, y):
        y_pred = model.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        mse_per_output = []
        for i in range(y.shape[1]):
            mse_per_output.append(mean_squared_error(y[:, i], y_pred[:, i]))
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mse_per_output': mse_per_output
        }
    
    def train(self, data_file="data/demonstrations.json", test_size=0.2, 
              models_to_train=['knn', 'rf', 'mlp']):
        print("\n=== 开始训练 ===")
        
        X, y = self.load_data(data_file)
        if X is None:
            return False
        
        if len(X) < 10:
            print(f"[错误] 数据太少: {len(X)} 个样本")
            return False
        
        X, y = self.normalize_data(X, y)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"[数据分割] 训练: {len(X_train)}, 验证: {len(X_val)}")
        
        results = {}
        
        if 'knn' in models_to_train:
            model, score = self.train_knn(X_train, y_train, X_val, y_val)
            results['knn'] = {'model': model, 'score': score}
        
        if 'rf' in models_to_train:
            model, score = self.train_random_forest(X_train, y_train, X_val, y_val)
            results['rf'] = {'model': model, 'score': score}
        
        if 'mlp' in models_to_train:
            model, score = self.train_mlp(X_train, y_train, X_val, y_val)
            results['mlp'] = {'model': model, 'score': score}
        
        if 'gb' in models_to_train:
            model, score = self.train_gboost(X_train, y_train, X_val, y_val)
            results['gb'] = {'model': model, 'score': score}
        
        for name, result in results.items():
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_model = name
                self.models['best'] = result['model']
        
        print(f"\n[Best model] {self.best_model} (R2={self.best_score:.4f})")
        
        return True
    
    def save(self, model_name=None):
        if model_name is None:
            model_name = self.best_model or "best"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(self.model_dir, f"{model_name}_{timestamp}.pkl")
        
        save_data = {
            'model': self.models.get('best'),
            'scalers': self.scalers,
            'model_type': self.best_model,
            'score': self.best_score,
            'timestamp': timestamp
        }
        
        joblib.dump(save_data, model_path)
        print(f"[模型保存] {model_path}")
        
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        metadata = {
            'model_path': model_path,
            'model_type': self.best_model,
            'score': self.best_score,
            'timestamp': timestamp
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def load(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.model_dir, "metadata.json")
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    metadata = json.load(f)
                model_path = metadata.get('model_path')
            else:
                print("[错误] 没有已保存的模型")
                return False
        
        if not os.path.exists(model_path):
            print(f"[错误] 模型文件不存在: {model_path}")
            return False
        
        save_data = joblib.load(model_path)
        
        self.models['best'] = save_data['model']
        self.scalers = save_data['scalers']
        self.best_model = save_data['model_type']
        self.best_score = save_data['score']
        
        print(f"[Model loaded] {self.best_model}, R2={self.best_score:.4f}")
        return True
    
    def predict(self, keypoints):
        if self.models.get('best') is None:
            print("[错误] 模型未加载")
            return None
        
        if 'X' not in self.scalers:
            print("[错误] 标准化器未加载")
            return None
        
        X = np.array(keypoints, dtype=np.float32).reshape(1, -1)
        
        X = self.scalers['X'].transform(X)
        
        y = self.models['best'].predict(X)
        
        y = self.scalers['y'].inverse_transform(y)
        
        return {
            'base': int(np.clip(y[0][0], 0, 180)),
            'arm1': int(np.clip(y[0][1], 0, 180)),
            'arm2': int(np.clip(y[0][2], 0, 180)),
            'gripper': int(np.clip(y[0][3], 0, 180)),
            'rotation': int(np.clip(y[0][4], 0, 180))
        }


def train_command():
    import argparse
    
    parser = argparse.ArgumentParser(description='训练机械臂控制模型')
    parser.add_argument('--data', default='data/demonstrations.json', help='数据文件')
    parser.add_argument('--models', nargs='+', default=['knn', 'rf', 'mlp'], 
                      help='要训练的模型')
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    success = trainer.train(args.data, args.test_size, args.models)
    
    if success:
        trainer.save()
    else:
        print("训练失败")


def interactive_train():
    print("\n=== 模型训练 ===")
    print("1. 开始训练 (KNN, RF, MLP)")
    print("2. 仅训练 KNN")
    print("3. 仅训练 Random Forest")
    print("4. 仅训练 MLP")
    print("5. 加载已有模型")
    print("q. 退出")
    
    choice = input("选择: ").strip()
    
    trainer = ModelTrainer()
    
    if choice == '1':
        trainer.train('data/demonstrations.json', 0.2, ['knn', 'rf', 'mlp'])
    elif choice == '2':
        trainer.train('data/demonstrations.json', 0.2, ['knn'])
    elif choice == '3':
        trainer.train('data/demonstrations.json', 0.2, ['rf'])
    elif choice == '4':
        trainer.train('data/demonstrations.json', 0.2, ['mlp'])
    elif choice == '5':
        trainer.load()
    elif choice == 'q':
        return
    
    if trainer.best_model:
        trainer.save()
    else:
        print("没有训练模型")


if __name__ == "__main__":
    interactive_train()