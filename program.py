import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

# MEMBACA DATASET
file_dataset = "data_bersih.csv"

df = pd.read_csv(file_dataset)

print("====================================")
print(" DATASET BERHASIL DIMUAT ")
print("====================================")

print(df.head())

# MEMBERSIHKAN SPASI DAN TAB
for kolom in df.columns:

    if df[kolom].dtype == object:

        df[kolom] = df[kolom].astype(str).str.strip()

# ENCODING DATA KATEGORI
encoding = {

    "normal": 0,
    "abnormal": 1,

    "present": 1,
    "notpresent": 0,

    "yes": 1,
    "no": 0,

    "poor": 1,
    "good": 0,

    "ckd": 1,
    "notckd": 0
}

df.replace(encoding, inplace=True)

# MENGHAPUS KOLOM ID
if "id" in df.columns:

    df.drop("id", axis=1, inplace=True)

# MEMILIH FITUR YANG DIGUNAKAN
fitur_digunakan = [

    "age",     # umur
    "bp",      # tekanan darah
    "bgr",     # gula darah
    "bu",      # urea darah
    "hemo",    # hemoglobin

    "htn",     # hipertensi
    "dm",      # diabetes
    "appet",   # nafsu makan buruk
    "pe",      # pembengkakan kaki
    "ane"      # anemia
]

# FITUR DAN TARGET
X = df[fitur_digunakan]

y = df["classification"]

# MENGUBAH TARGET MENJADI INTEGER
y = y.astype(int)

# MEMBAGI DATA TRAINING DAN TESTING
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# NORMALISASI DATA
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# MEMBUAT MODEL RANDOM FOREST
model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)

# TRAINING MODEL
model.fit(X_train, y_train)

# PREDIKSI DATA TEST
prediksi = model.predict(X_test)

# EVALUASI MODEL
akurasi = accuracy_score(y_test, prediksi)
r2 = r2_score(y_test, prediksi)

print("\n====================================")
print(" HASIL EVALUASI MODEL ")
print("====================================")

print(f"Akurasi Model : {akurasi * 100:.2f}%")
print(f"Akurasi model (R2 Score): {r2:.4f}")

# FUNGSI MENENTUKAN TINGKAT KEPARAHAN
def tentukan_stage(bu, hemo, total_gejala):

    if bu < 40 and hemo > 12:

        return "Stage 1 - Ringan"

    elif bu < 80:

        return "Stage 2 - Sedang"

    elif bu < 120:

        return "Stage 3 - Berat"

    else:

        return "Stage 4 - Sangat Berat"

# INPUT USER
print("\n====================================")
print(" SISTEM DIAGNOSA PENYAKIT GINJAL ")
print("====================================")

age = float(input("Umur : "))
bp = float(input("Tekanan darah : "))
print (f"normal gula darah 70-140 mg/dl")
bgr = float(input("Gula darah : "))
print(f"normal urea darah 7- 40 mg/dl")
bu = float(input("Urea darah : "))
print(f"13- 17g/dl")
hemo = float(input("Hemoglobin : "))

print("\nInput 1 untuk YA dan 0 untuk TIDAK")
htn = int(input("Tekanan darah tinggi : "))
dm = int(input("Diabetes : "))
appet = int(input("Nafsu makan buruk : "))
pe = int(input("Pembengkakan kaki : "))
ane = int(input("Anemia : "))

# MEMBENTUK INPUT USER
input_user = pd.DataFrame([[
    age,
    bp,
    bgr,
    bu,
    hemo,

    htn,
    dm,
    appet,
    pe,
    ane

]], columns=fitur_digunakan)

# NORMALISASI INPUT USER
input_user = scaler.transform(input_user)

# PREDIKSI USER
hasil = model.predict(input_user)

probabilitas = model.predict_proba(input_user)

# MENGHITUNG TOTAL GEJALA
total_gejala = htn + dm + appet + pe + ane

# MENGHITUNG PERSENTASE CKD
persentase_ckd = probabilitas[0][1] * 100

# HASIL DIAGNOSA
print("\n====================================")
print(" HASIL DIAGNOSA ")
print("====================================")

# VALIDASI LOGIKA AGAR LEBIH REALISTIS
if (
    total_gejala <= 1
    and bp < 140
    and bgr < 140
    and bu < 40
    and hemo >= 12
):

    print("Prediksi Penyakit : Tidak Terindikasi Penyakit Ginjal Kronis")

    print("Tingkat Risiko : Rendah")

    print(f"Probabilitas CKD : {persentase_ckd:.2f}%")

else:

    print("Prediksi Penyakit : Penyakit Ginjal Kronis (CKD)")

    stage = tentukan_stage(bu, hemo, total_gejala)

    print(f"Tingkat Keparahan : {stage}")

    print(f"Probabilitas CKD : {persentase_ckd:.2f}%")