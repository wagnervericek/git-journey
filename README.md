# House Prices - Advanced Regression Techniques

Kaggle'ın klasik ev fiyatı tahmini yarışması için uçtan uca bir makine öğrenmesi pipeline'ı. Ames, Iowa'daki 1460 evin 79 farklı özelliğinden (büyüklük, kalite, konum, yaş vb.) satış fiyatını tahmin ediyoruz.

🔗 [Yarışma sayfası](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Sonuç

| Metrik | Değer |
|---|---|
| Kaggle Public Score (RMSE, log ölçeğinde) | **0.12546** |
| En iyi model | CatBoost |
| Başlangıç skoru | 0.13271 |
| Toplam iyileşme | %5.5 |

## Pipeline

### 1. Veri Analizi (EDA)
- Hedef değişken (`SalePrice`) sağa çarpık dağılım gösteriyor → `log1p` dönüşümü ile normalize edildi
- Korelasyon analizi: `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64) en güçlü sinyaller
- Görsel + istatistiksel (residual bazlı) outlier tespiti

### 2. Veri Temizleme
- **"Yok" anlamına gelen eksik veri** (`PoolQC`, `GarageType`, `FireplaceQu` vb.) → `"None"` kategorisiyle dolduruldu
- **Gerçek eksik veri** (`LotFrontage`, `Electrical` vb.) → median/mode ile dolduruldu
- Residual bazlı outlier temizliği: modelin en çok yanıldığı %1'lik dilim (12 satır) çıkarıldı — genelde `SaleCondition=Abnorml/Family/Partial` gibi piyasa dışı satışlar

### 3. Feature Engineering
- `TotalSF` (toplam kullanılabilir alan), `HouseAge`, `TotalBath`, `YearsSinceRemodel`
- `NeighborhoodRank` — mahalleleri ortalama fiyata göre sıralayan ordinal encoding
- Denenip işe yaramayan fikirler: domain etkileşim terimleri (`Qual × SF`), Box-Cox dönüşümü (ağaç modellerinde etkisiz), agresif feature selection

### 4. Modelleme
Denenen modeller ve sonuçları (kendi test setimizde R²):

| Model | R² |
|---|---|
| Linear Regression | 0.888 |
| Random Forest (tuned) | ~0.90 |
| LightGBM | 0.906 |
| XGBoost (tuned) | 0.913-0.917 |
| PyTorch (basit NN) | 0.867-0.913 |
| **CatBoost** | **0.919** |

CatBoost kazandı — kategorik değişkenleri one-hot encoding'e gerek kalmadan native olarak işleyebilmesi büyük avantaj sağladı.

### 5. Denenen Ensemble Teknikleri
- Basit blend (ortalama), ağırlıklı blend
- StackingRegressor (XGBoost + LightGBM + Linear/Ridge)
- Seed ensembling (aynı model, farklı random state)

**Not:** Bu tekniklerin hiçbiri gizli test setinde anlamlı bir iyileşme sağlamadı — bazıları kendi test setimizde iyi görünse de Kaggle skorunda nötr/hafif negatif çıktı. En büyük kazanç, karmaşık ensemble'lardan değil, **doğru tekil modeli seçmekten** (CatBoost) ve **outlier temizliğinden** geldi.

## Kullanılan Kütüphaneler
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`, `matplotlib`, `seaborn`

## Öğrenilen Dersler
- Ağaç tabanlı modeller (XGBoost, CatBoost) feature'ların ölçeğine/dağılımına duyarsız — log/Box-Cox dönüşümleri bu modellerde etkisiz kalabilir
- Cross-validation'daki küçük iyileşmeler her zaman gizli test setine yansımıyor
- Ensemble/stacking her zaman iyileştirmez — zayıf bir modeli ensemble'a katmak güçlü modelleri de aşağı çekebilir
- CatBoost'un native kategorik değişken desteği, yüksek kardinaliteli kategorik verilerde one-hot encoding'e göre gerçek bir avantaj sağlayabilir
