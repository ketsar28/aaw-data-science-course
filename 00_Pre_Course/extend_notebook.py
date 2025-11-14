#!/usr/bin/env python3
"""
Extend notebook dengan Part 3, 4, 5, 6
"""
import nbformat as nbf

# Load existing notebook
with open('00_python_crash_course.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Get existing cells
cells = nb['cells']

# ============================================================================
# PART 3: DATA STRUCTURES
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 📦 PART 3: Data Structures (60 menit)

## Progress: 3/6 🟩🟩🟩⬜⬜⬜

Data Structures = Cara menyimpan dan mengorganisir banyak data

---

## 3.1 List - Daftar/Array

List adalah kumpulan item yang ter-urut dan bisa diubah (mutable).

**Analogi**: Seperti daftar belanjaan. Bisa tambah item, hapus item, ubah item.
"""))

cells.append(nbf.v4.new_code_cell("""# Membuat list
buah = ["apel", "jeruk", "mangga", "pisang"]
angka = [1, 2, 3, 4, 5]
campur = ["budi", 25, True, 3.14]  # Bisa campur tipe data

print("List buah:", buah)
print("List angka:", angka)
print("List campur:", campur)"""))

cells.append(nbf.v4.new_code_cell("""# Mengakses elemen list dengan index
# Index dimulai dari 0!
buah = ["apel", "jeruk", "mangga", "pisang"]

print("Buah pertama:", buah[0])    # apel
print("Buah kedua:", buah[1])      # jeruk
print("Buah terakhir:", buah[-1])  # pisang (index negatif dari belakang)
print("Buah kedua dari belakang:", buah[-2])  # mangga"""))

cells.append(nbf.v4.new_code_cell("""# Slicing - mengambil sebagian list
angka = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print("Elemen 0-3:", angka[0:4])    # [0, 1, 2, 3] - tidak termasuk index 4
print("Elemen 5-8:", angka[5:9])    # [5, 6, 7, 8]
print("Dari awal sampai index 4:", angka[:5])  # [0, 1, 2, 3, 4]
print("Dari index 6 sampai akhir:", angka[6:]) # [6, 7, 8, 9]
print("Semua elemen:", angka[:])    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"""))

cells.append(nbf.v4.new_markdown_cell("""### Method-method List

List punya banyak method (fungsi) yang berguna:
"""))

cells.append(nbf.v4.new_code_cell("""# append() - Menambah elemen di akhir
buah = ["apel", "jeruk"]
print("Awal:", buah)

buah.append("mangga")
print("Setelah append mangga:", buah)

buah.append("pisang")
print("Setelah append pisang:", buah)"""))

cells.append(nbf.v4.new_code_cell("""# insert() - Menambah elemen di posisi tertentu
buah = ["apel", "jeruk", "pisang"]
print("Awal:", buah)

buah.insert(1, "mangga")  # Insert di index 1
print("Setelah insert mangga di index 1:", buah)"""))

cells.append(nbf.v4.new_code_cell("""# remove() - Menghapus elemen berdasarkan nilai
buah = ["apel", "jeruk", "mangga", "pisang"]
print("Awal:", buah)

buah.remove("jeruk")
print("Setelah remove jeruk:", buah)"""))

cells.append(nbf.v4.new_code_cell("""# pop() - Menghapus dan return elemen di index tertentu
buah = ["apel", "jeruk", "mangga", "pisang"]
print("Awal:", buah)

hapus = buah.pop(2)  # Hapus index 2
print("Elemen yang dihapus:", hapus)
print("Setelah pop:", buah)

terakhir = buah.pop()  # Tanpa parameter = hapus elemen terakhir
print("Elemen terakhir yang dihapus:", terakhir)
print("Akhir:", buah)"""))

cells.append(nbf.v4.new_code_cell("""# len() - Panjang list
buah = ["apel", "jeruk", "mangga", "pisang"]
print("Jumlah buah:", len(buah))

# in - Cek apakah elemen ada dalam list
print("Apakah ada apel?", "apel" in buah)
print("Apakah ada semangka?", "semangka" in buah)"""))

cells.append(nbf.v4.new_code_cell("""# sort() - Mengurutkan list
angka = [5, 2, 8, 1, 9, 3]
print("Sebelum sort:", angka)

angka.sort()
print("Setelah sort (ascending):", angka)

angka.sort(reverse=True)
print("Setelah sort (descending):", angka)"""))

cells.append(nbf.v4.new_code_cell("""# List Comprehension - Cara singkat membuat list
# Format: [expression for item in iterable]

# Cara biasa:
kuadrat = []
for i in range(1, 6):
    kuadrat.append(i ** 2)
print("Cara biasa:", kuadrat)

# Dengan list comprehension:
kuadrat2 = [i**2 for i in range(1, 6)]
print("List comprehension:", kuadrat2)"""))

cells.append(nbf.v4.new_code_cell("""# List comprehension dengan kondisi
# Ambil hanya angka genap dari 1-20
genap = [i for i in range(1, 21) if i % 2 == 0]
print("Angka genap 1-20:", genap)

# Kuadrat dari angka ganjil 1-10
kuadrat_ganjil = [i**2 for i in range(1, 11) if i % 2 != 0]
print("Kuadrat angka ganjil 1-10:", kuadrat_ganjil)"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 3.2 Dictionary - Key-Value Pairs

Dictionary menyimpan data dalam pasangan **key: value**.

**Analogi**: Seperti kamus. Anda mencari kata (key) untuk mendapat definisi (value).
"""))

cells.append(nbf.v4.new_code_cell("""# Membuat dictionary
siswa = {
    "nama": "Budi",
    "umur": 17,
    "kelas": "12 IPA",
    "nilai": 85
}

print("Dictionary siswa:", siswa)
print("Tipe data:", type(siswa))"""))

cells.append(nbf.v4.new_code_cell("""# Mengakses value dengan key
siswa = {"nama": "Budi", "umur": 17, "kelas": "12 IPA"}

print("Nama:", siswa["nama"])
print("Umur:", siswa["umur"])

# Cara lebih aman dengan get() (tidak error jika key tidak ada)
print("Nilai:", siswa.get("nilai"))  # None (key tidak ada)
print("Nilai:", siswa.get("nilai", 0))  # 0 (default value jika tidak ada)"""))

cells.append(nbf.v4.new_code_cell("""# Menambah/mengubah value
siswa = {"nama": "Budi", "umur": 17}
print("Awal:", siswa)

siswa["kelas"] = "12 IPA"  # Tambah key baru
print("Setelah tambah kelas:", siswa)

siswa["umur"] = 18  # Ubah value yang sudah ada
print("Setelah ubah umur:", siswa)"""))

cells.append(nbf.v4.new_code_cell("""# Method-method dictionary
siswa = {"nama": "Budi", "umur": 17, "kelas": "12 IPA"}

# keys() - Ambil semua keys
print("Keys:", list(siswa.keys()))

# values() - Ambil semua values
print("Values:", list(siswa.values()))

# items() - Ambil semua pasangan (key, value)
print("Items:", list(siswa.items()))"""))

cells.append(nbf.v4.new_code_cell("""# Loop dictionary
siswa = {"nama": "Budi", "umur": 17, "kelas": "12 IPA", "nilai": 85}

# Loop keys
print("Loop keys:")
for key in siswa:
    print(f"  {key}: {siswa[key]}")

print("\\nLoop items:")
for key, value in siswa.items():
    print(f"  {key}: {value}")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 3.3 Tuple - List yang Tidak Bisa Diubah

Tuple seperti list, tapi **immutable** (tidak bisa diubah setelah dibuat).

**Analogi**: Seperti foto - sekali jadi, tidak bisa diubah. Kalau mau beda, buat foto baru.
"""))

cells.append(nbf.v4.new_code_cell("""# Membuat tuple (pakai kurung biasa atau tanpa kurung)
koordinat = (3, 4)
warna = ("merah", "hijau", "biru")
data = 1, 2, 3, 4, 5  # Bisa tanpa kurung

print("Tuple koordinat:", koordinat)
print("Tuple warna:", warna)
print("Tuple data:", data)
print("Tipe:", type(koordinat))"""))

cells.append(nbf.v4.new_code_cell("""# Mengakses elemen tuple (sama seperti list)
warna = ("merah", "hijau", "biru")
print("Warna pertama:", warna[0])
print("Warna terakhir:", warna[-1])"""))

cells.append(nbf.v4.new_code_cell("""# Tuple tidak bisa diubah!
# warna[0] = "kuning"  # Ini akan ERROR!

# Tapi bisa buat tuple baru
warna = ("merah", "hijau", "biru")
warna_baru = ("kuning",) + warna[1:]  # Ganti elemen pertama
print("Warna baru:", warna_baru)"""))

cells.append(nbf.v4.new_code_cell("""# Tuple unpacking - Assign ke beberapa variable sekaligus
koordinat = (3, 4)
x, y = koordinat  # Unpacking
print("x =", x)
print("y =", y)

# Swap values dengan tuple
a = 5
b = 10
print(f"Sebelum swap: a={a}, b={b}")

a, b = b, a  # Swap!
print(f"Setelah swap: a={a}, b={b}")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 3.4 Set - Kumpulan Unik tanpa Duplikat

Set adalah kumpulan elemen **unik** (tidak ada duplikat) dan **tidak berurutan**.

**Analogi**: Seperti keanggotaan klub - seseorang hanya bisa jadi anggota sekali (tidak duplikat).
"""))

cells.append(nbf.v4.new_code_cell("""# Membuat set
angka = {1, 2, 3, 4, 5}
buah = {"apel", "jeruk", "mangga"}

print("Set angka:", angka)
print("Set buah:", buah)

# Set otomatis hapus duplikat
angka_duplikat = {1, 2, 2, 3, 3, 3, 4, 5, 5}
print("Set dengan duplikat:", angka_duplikat)  # Duplikat hilang!"""))

cells.append(nbf.v4.new_code_cell("""# Operasi set
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# Union (gabungan)
print("A union B:", A | B)  # atau A.union(B)

# Intersection (irisan/yang sama)
print("A intersection B:", A & B)  # atau A.intersection(B)

# Difference (selisih)
print("A - B:", A - B)  # yang ada di A tapi tidak di B
print("B - A:", B - A)  # yang ada di B tapi tidak di A"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 🏋️ Latihan Part 3

### Latihan 3.1: List Operations
Diberikan list: `nilai = [78, 92, 85, 67, 95, 73, 88]`

1. Hitung rata-rata nilai
2. Temukan nilai tertinggi dan terendah
3. Hitung berapa banyak siswa yang lulus (nilai >= 75)
4. Buat list baru berisi hanya nilai yang lulus (>= 75)
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
nilai = [78, 92, 85, 67, 95, 73, 88]


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 3.2: Dictionary
Buat dictionary untuk menyimpan data 3 produk (nama, harga, stok).
Lalu hitung total value inventory (harga × stok untuk semua produk).

Contoh:
```
produk1: nama="Laptop", harga=5000000, stok=10
produk2: nama="Mouse", harga=100000, stok=50
...
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 3.3: Set Operations
Diberikan dua set:
- `mata_kuliah_A = {"Matematika", "Fisika", "Kimia", "Biologi"}`
- `mata_kuliah_B = {"Fisika", "Kimia", "Sejarah", "Geografi"}`

Temukan:
1. Mata kuliah yang ada di kedua kelas
2. Mata kuliah yang hanya ada di kelas A
3. Semua mata kuliah yang ada (union)
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
mata_kuliah_A = {"Matematika", "Fisika", "Kimia", "Biologi"}
mata_kuliah_B = {"Fisika", "Kimia", "Sejarah", "Geografi"}


"""))

cells.append(nbf.v4.new_markdown_cell("""---

✅ **Part 3 Selesai!** Anda sudah paham data structures!

**Next**: Part 4 - Functions

---"""))

# Simpan updated notebook
nb['cells'] = cells
with open('00_python_crash_course.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✅ Notebook extended! Total cells: {len(cells)}")
