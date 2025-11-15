#!/usr/bin/env python3
"""Create 00_exercises.ipynb dengan 20 latihan"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("""# 🏋️ Python Crash Course - Latihan & Exercises

---

## 📋 Tentang File Ini

File ini berisi **20 latihan** untuk mempraktikkan semua yang sudah Anda pelajari di `00_python_crash_course.ipynb`.

### 🎯 Cara Menggunakan:
1. **Baca** soal dengan teliti
2. **Coba solve sendiri** dulu (minimal 15-20 menit)
3. **Jika stuck**, lihat **Hint** yang disediakan
4. **Jika masih stuck**, buka `00_solutions.ipynb`
5. **PENTING**: Pahami solution, jangan hanya copy-paste!

### 📊 Tingkat Kesulitan:
- 🟢 Easy (Soal 1-7)
- 🟡 Medium (Soal 8-15)
- 🔴 Hard (Soal 16-20)

---

**Good luck & have fun coding!** 💪

---"""))

# ============================================================================
# LATIHAN 1-7: EASY (Basics, Variables, Operators)
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🟢 LATIHAN 1: Print & Variables (Easy)

**Soal**:
Buat program yang menyimpan informasi diri Anda (nama, umur, kota asal, hobi) dalam variables, 
lalu print dalam format yang rapi seperti ini:

```
=== Perkenalan ===
Nama: [nama Anda]
Umur: [umur] tahun
Kota: [kota]
Hobi: [hobi]
=================
```

### 💡 Hint:
- Gunakan 4 variables
- Gunakan f-string atau .format() untuk formatting yang rapi
- Bisa pakai \\n untuk newline

### ✅ Expected Output:
```
=== Perkenalan ===
Nama: Budi Santoso
Umur: 25 tahun
Kota: Jakarta
Hobi: Membaca
=================
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini





"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟢 LATIHAN 2: Operasi Matematika (Easy)

**Soal**:
Sebuah toko membeli 50 barang dengan harga @Rp 15.000 per barang.
Toko menjual dengan harga @Rp 20.000 per barang.

Hitung:
1. Total modal (biaya beli semua barang)
2. Total pendapatan (jika semua terjual)
3. Total keuntungan
4. Persentase keuntungan (keuntungan / modal × 100)

Print semua hasil dengan format yang rapi.

### 💡 Hint:
- Gunakan variables untuk menyimpan nilai
- Hitung step by step
- Format angka dengan f-string: f"{angka:,}" untuk pemisah ribuan

### ✅ Expected Output:
```
Total Modal: Rp 750,000
Total Pendapatan: Rp 1,000,000
Keuntungan: Rp 250,000
Persentase Keuntungan: 33.33%
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini





"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟢 LATIHAN 3: Konversi Suhu (Easy)

**Soal**:
Buat program untuk mengkonversi suhu dari Celsius ke Fahrenheit dan Kelvin.

Formula:
- Fahrenheit = (Celsius × 9/5) + 32
- Kelvin = Celsius + 273.15

Test dengan Celsius = 37 (suhu tubuh manusia).

### 💡 Hint:
- Buat variable untuk celsius
- Hitung fahrenheit dan kelvin
- Print hasilnya

### ✅ Expected Output:
```
37°C = 98.6°F
37°C = 310.15K
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini




"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟢 LATIHAN 4: Kondisi If-Else (Easy)

**Soal**:
Buat program untuk menentukan kategori usia:
- 0-12: Anak-anak
- 13-17: Remaja
- 18-59: Dewasa
- 60+: Lansia

Test dengan berbagai nilai umur.

### 💡 Hint:
- Gunakan if-elif-else
- Perhatikan range angka

### ✅ Expected Output (untuk umur=15):
```
Umur 15 tahun: Kategori Remaja
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
umur = 15  # Ganti dengan nilai lain untuk testing





"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟢 LATIHAN 5: Loop Sederhana (Easy)

**Soal**:
Buat program yang mencetak tabel perkalian untuk angka tertentu (1-10).

Contoh untuk angka 5:
```
5 x 1 = 5
5 x 2 = 10
5 x 3 = 15
...
5 x 10 = 50
```

### 💡 Hint:
- Gunakan for loop dengan range(1, 11)
- Format output dengan f-string

### ✅ Expected Output: Lihat contoh di atas
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
angka = 5  # Ganti dengan angka lain untuk testing




"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟢 LATIHAN 6: List Basics (Easy)

**Soal**:
Diberikan list nilai ujian: `[85, 92, 78, 67, 95, 73, 88, 82, 90, 76]`

Hitung dan print:
1. Jumlah total nilai
2. Nilai rata-rata
3. Nilai tertinggi
4. Nilai terendah
5. Berapa banyak siswa yang lulus (nilai >= 75)

### 💡 Hint:
- Gunakan built-in functions: sum(), len(), max(), min()
- Untuk menghitung yang lulus: loop + counter atau filter

### ✅ Expected Output:
```
Total Nilai: 826
Rata-rata: 82.6
Nilai Tertinggi: 95
Nilai Terendah: 67
Jumlah Lulus (>=75): 8 siswa
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
nilai = [85, 92, 78, 67, 95, 73, 88, 82, 90, 76]






"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟢 LATIHAN 7: Dictionary Basics (Easy)

**Soal**:
Buat dictionary untuk menyimpan data buku:
- Judul: "Belajar Python"
- Pengarang: "John Doe"
- Tahun: 2023
- Halaman: 350
- Harga: 150000

Lalu:
1. Print semua informasi buku
2. Update harga menjadi 135000 (diskon 10%)
3. Tambahkan key "rating" dengan value 4.5
4. Print dictionary yang sudah diupdate

### 💡 Hint:
- Buat dictionary dengan key-value pairs
- Update: dict[key] = new_value
- Tambah: dict[new_key] = value

### ✅ Expected Output: Dictionary dengan informasi lengkap
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini





"""))

# ============================================================================
# LATIHAN 8-15: MEDIUM (Kombinasi konsep)
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""---

# 🟡 LATIHAN 8: FizzBuzz (Medium)

**Soal**:
Buat program FizzBuzz untuk angka 1-100:
- Jika angka habis dibagi 3: print "Fizz"
- Jika angka habis dibagi 5: print "Buzz"
- Jika angka habis dibagi 3 DAN 5: print "FizzBuzz"
- Selainnya: print angkanya

### 💡 Hint:
- Gunakan loop for
- Cek kondisi dari yang paling spesifik (3 dan 5) ke yang umum
- Gunakan operator modulus (%)

### ✅ Expected Output (contoh):
```
1
2
Fizz
4
Buzz
Fizz
7
8
Fizz
Buzz
11
Fizz
13
14
FizzBuzz
...
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini




"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟡 LATIHAN 9: Palindrome Checker (Medium)

**Soal**:
Buat function `is_palindrome(kata)` yang mengecek apakah sebuah kata adalah palindrome.

Palindrome = kata yang sama jika dibaca dari depan atau belakang.
Contoh: "katak", "radar", "kodok", "level"

Function harus return True/False.

Test dengan: "katak", "python", "radar", "hello"

### 💡 Hint:
- Palindrome: kata == kata[::-1] (reversed)
- Bisa juga dengan loop dan compare index

### ✅ Expected Output:
```
katak: True
python: False
radar: True
hello: False
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
def is_palindrome(kata):
    # Tulis kode function di sini
    pass

# Test cases
test_words = ["katak", "python", "radar", "hello"]




"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟡 LATIHAN 10: List Comprehension (Medium)

**Soal**:
Dari list angka 1-100, buat 3 list baru menggunakan **list comprehension**:

1. `kuadrat`: Berisi kuadrat dari semua angka
2. `genap`: Berisi hanya angka genap
3. `kelipatan_3_5`: Berisi angka yang merupakan kelipatan 3 ATAU 5

Lalu print:
- 10 angka pertama dari masing-masing list
- Total elemen di setiap list

### 💡 Hint:
- Format: `[expression for item in iterable if condition]`
- Kondisi genap: `i % 2 == 0`
- Kondisi kelipatan 3 atau 5: `i % 3 == 0 or i % 5 == 0`

### ✅ Expected Output:
```
Kuadrat (10 pertama): [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
Genap (10 pertama): [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
Kelipatan 3 atau 5 (10 pertama): [3, 5, 6, 9, 10, 12, 15, 18, 20, 21]

Total - Kuadrat: 100, Genap: 50, Kelipatan 3/5: 54
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini




"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟡 LATIHAN 11: Function dengan Multiple Return (Medium)

**Soal**:
Buat function `analisis_list(angka_list)` yang mengembalikan:
1. Total (sum)
2. Rata-rata
3. Maksimum
4. Minimum
5. Median (nilai tengah setelah diurutkan)

Test dengan: `[45, 23, 78, 12, 90, 34, 67, 89, 56, 41]`

### 💡 Hint:
- Return multiple values sebagai tuple
- Untuk median: sort list, ambil element di tengah
- Jika jumlah elemen genap, median = rata-rata 2 angka tengah

### ✅ Expected Output:
```
Total: 535
Rata-rata: 53.5
Maksimum: 90
Minimum: 12
Median: 50.5
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
def analisis_list(angka_list):
    # Tulis kode function di sini
    pass

# Test
angka = [45, 23, 78, 12, 90, 34, 67, 89, 56, 41]





"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟡 LATIHAN 12: Dictionary dari Lists (Medium)

**Soal**:
Diberikan 3 lists:
- `nama = ["Budi", "Ani", "Citra", "Dedi"]`
- `nilai_math = [85, 92, 78, 88]`
- `nilai_english = [78, 85, 92, 82]`

Buat list of dictionaries dimana setiap dictionary berisi:
- nama siswa
- nilai math
- nilai english
- rata-rata kedua nilai

Lalu print informasi setiap siswa dalam format yang rapi.

### 💡 Hint:
- Gunakan zip() untuk menggabungkan lists
- Loop dan buat dictionary untuk setiap siswa
- Append ke list of dictionaries

### ✅ Expected Output:
```
Siswa 1: Budi
  Math: 85, English: 78, Rata-rata: 81.5

Siswa 2: Ani
  Math: 92, English: 85, Rata-rata: 88.5
...
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
nama = ["Budi", "Ani", "Citra", "Dedi"]
nilai_math = [85, 92, 78, 88]
nilai_english = [78, 85, 92, 82]





"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟡 LATIHAN 13: Nested Loop - Pattern (Medium)

**Soal**:
Buat program yang mencetak pattern angka seperti ini:

```
1
1 2
1 2 3
1 2 3 4
1 2 3 4 5
1 2 3 4
1 2 3
1 2
1
```

### 💡 Hint:
- Perlu 2 loops: satu untuk naik, satu untuk turun
- Loop pertama: i dari 1 sampai 5
- Loop kedua: i dari 5 sampai 1
- Di dalam setiap loop, nested loop untuk print angka

### ✅ Expected Output: Lihat pattern di atas
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini




"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟡 LATIHAN 14: File Operations (Medium)

**Soal**:
1. Buat file "data_siswa.txt" yang berisi nama dan nilai siswa (format: nama,nilai)
2. Baca file tersebut
3. Hitung rata-rata nilai
4. Tampilkan siswa dengan nilai di atas rata-rata
5. Simpan hasilnya ke file baru "siswa_terbaik.txt"

Data siswa:
```
Budi,85
Ani,92
Citra,78
Dedi,88
Eka,95
Fani,73
Gita,82
```

### 💡 Hint:
- Write file dengan loop
- Read file, split dengan koma
- Convert nilai ke int
- Hitung rata-rata, filter yang di atas rata-rata
- Write hasil ke file baru

### ✅ Expected Output: File dengan siswa nilai > rata-rata
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini





"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🟡 LATIHAN 15: Try-Except dengan User Input (Medium)

**Soal**:
Buat kalkulator sederhana yang:
1. Minta user input 2 angka
2. Minta user pilih operasi (+, -, *, /)
3. Hitung dan tampilkan hasil
4. Handle semua error yang mungkin terjadi:
   - Input bukan angka
   - Pembagian dengan nol
   - Operasi tidak valid

Program harus tetap jalan dan memberikan pesan error yang jelas.

### 💡 Hint:
- Gunakan try-except untuk setiap input
- Gunakan if-elif untuk operasi
- ValueError untuk input bukan angka
- ZeroDivisionError untuk pembagian nol

### ✅ Expected Output:
```
Angka 1: 10
Angka 2: 0
Operasi (+,-,*,/): /
Error: Tidak bisa membagi dengan nol!
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
# Simulasi user input (ganti dengan nilai lain untuk testing)
input_1 = "10"
input_2 = "0"
operasi = "/"







"""))

# ============================================================================
# LATIHAN 16-20: HARD (Complex problems)
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""---

# 🔴 LATIHAN 16: Prime Numbers (Hard)

**Soal**:
Buat function `cari_bilangan_prima(n)` yang mengembalikan list semua bilangan prima dari 2 sampai n.

Bilangan prima = bilangan yang hanya habis dibagi 1 dan dirinya sendiri.

Contoh: 2, 3, 5, 7, 11, 13, ...

Test dengan n = 50

### 💡 Hint:
- Function `is_prime(num)` untuk cek apakah angka prima
- Loop dari 2 sampai num-1, cek apakah ada yang habis membagi num
- Jika tidak ada, angka tersebut prima
- Gunakan function is_prime dalam loop 2 sampai n

### ✅ Expected Output:
```
Bilangan prima sampai 50:
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini




"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🔴 LATIHAN 17: Fibonacci Sequence (Hard)

**Soal**:
Buat 2 function:
1. `fibonacci_iterative(n)` - Hitung fibonacci ke-n dengan iterasi
2. `fibonacci_recursive(n)` - Hitung fibonacci ke-n dengan rekursi

Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
Rumus: F(n) = F(n-1) + F(n-2), dengan F(0)=0, F(1)=1

Print 20 angka fibonacci pertama menggunakan kedua metode.

### 💡 Hint:
- Iterative: Gunakan loop dengan 2 variables (prev, current)
- Recursive: Function memanggil dirinya sendiri dengan n-1 dan n-2
- Base case: if n <= 1, return n

### ✅ Expected Output:
```
20 Fibonacci pertama (Iterative):
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181

20 Fibonacci pertama (Recursive):
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini




"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🔴 LATIHAN 18: Frequency Counter (Hard)

**Soal**:
Diberikan string panjang (paragraf), buat program untuk:
1. Hitung frekuensi setiap kata (case-insensitive)
2. Tampilkan 10 kata yang paling sering muncul
3. Tampilkan total kata unik
4. Hitung rata-rata panjang kata

Test dengan teks berikut:
```
Python adalah bahasa pemrograman yang mudah dipelajari. 
Python digunakan untuk data science machine learning dan web development.
Banyak perusahaan menggunakan Python karena Python sangat powerful dan mudah.
```

### 💡 Hint:
- Lower case semua text
- Split menjadi list kata
- Buat dictionary untuk frequency counter
- Sort dictionary by value (descending)
- Gunakan slicing untuk ambil top 10

### ✅ Expected Output:
```
10 Kata Paling Sering:
python: 4
mudah: 2
dan: 2
...

Total Kata Unik: X kata
Rata-rata Panjang Kata: X.XX karakter
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
text = \"\"\"
Python adalah bahasa pemrograman yang mudah dipelajari. 
Python digunakan untuk data science machine learning dan web development.
Banyak perusahaan menggunakan Python karena Python sangat powerful dan mudah.
\"\"\"






"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🔴 LATIHAN 19: Student Grade Management System (Hard)

**Soal**:
Buat sistem manajemen nilai siswa dengan fitur:

1. **Tambah Siswa**: Fungsi untuk menambah siswa baru dengan nama dan nilai (dict)
2. **Hitung Statistik**: Fungsi untuk menghitung statistik kelas:
   - Rata-rata nilai
   - Nilai tertinggi & terendah
   - Berapa banyak yang lulus (>= 75)
3. **Ranking**: Fungsi untuk menampilkan ranking siswa
4. **Simpan ke File**: Fungsi untuk save data ke CSV
5. **Baca dari File**: Fungsi untuk load data dari CSV

Implementasikan semua fungsi dan test dengan minimal 10 siswa.

### 💡 Hint:
- Gunakan dictionary atau list of dictionaries untuk menyimpan data
- Setiap function punya tanggung jawab spesifik
- Untuk ranking: sort berdasarkan nilai

### ✅ Expected Output:
```
=== Statistik Kelas ===
Rata-rata: 82.5
Tertinggi: 95 (Ani)
Terendah: 67 (Dedi)
Lulus: 8/10 siswa

=== Ranking ===
1. Ani - 95
2. Eka - 92
3. Budi - 88
...
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini









"""))

cells.append(nbf.v4.new_markdown_cell("""---

# 🔴 LATIHAN 20: Data Analyzer (Hard)

**Soal**:
Buat program untuk menganalisis data penjualan dengan fitur:

Data (dalam format list of dictionaries):
```python
sales = [
    {"tanggal": "2024-01-01", "produk": "Laptop", "jumlah": 2, "harga": 5000000},
    {"tanggal": "2024-01-01", "produk": "Mouse", "jumlah": 10, "harga": 100000},
    {"tanggal": "2024-01-02", "produk": "Laptop", "jumlah": 1, "harga": 5000000},
    # ... (minimal 20 data)
]
```

Hitung dan tampilkan:
1. **Total penjualan** (sum of jumlah × harga)
2. **Produk terlaris** (berdasarkan jumlah terjual)
3. **Produk dengan revenue tertinggi**
4. **Penjualan per hari**
5. **Rata-rata nilai transaksi**

Buat visualisasi sederhana dengan ASCII bar chart untuk top 3 produk.

### 💡 Hint:
- Gunakan dictionary untuk aggregate data
- Loop untuk menghitung totals
- sorted() dengan key untuk ranking
- ASCII bar chart: print "*" sebanyak nilai tertentu

### ✅ Expected Output:
```
=== Analisis Penjualan ===
Total Revenue: Rp 50,000,000

Produk Terlaris:
1. Mouse: 50 unit
2. Laptop: 10 unit
3. Keyboard: 30 unit

Top Revenue:
Laptop ********** (Rp 25,000,000)
Keyboard ******   (Rp 15,000,000)
Mouse ****        (Rp 10,000,000)
```
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini
sales = [
    {"tanggal": "2024-01-01", "produk": "Laptop", "jumlah": 2, "harga": 5000000},
    {"tanggal": "2024-01-01", "produk": "Mouse", "jumlah": 10, "harga": 100000},
    {"tanggal": "2024-01-02", "produk": "Laptop", "jumlah": 1, "harga": 5000000},
    {"tanggal": "2024-01-02", "produk": "Keyboard", "jumlah": 5, "harga": 500000},
    {"tanggal": "2024-01-03", "produk": "Mouse", "jumlah": 15, "harga": 100000},
    {"tanggal": "2024-01-03", "produk": "Laptop", "jumlah": 3, "harga": 5000000},
    {"tanggal": "2024-01-04", "produk": "Keyboard", "jumlah": 10, "harga": 500000},
    {"tanggal": "2024-01-04", "produk": "Mouse", "jumlah": 20, "harga": 100000},
    {"tanggal": "2024-01-05", "produk": "Laptop", "jumlah": 4, "harga": 5000000},
    {"tanggal": "2024-01-05", "produk": "Keyboard", "jumlah": 15, "harga": 500000},
]









"""))

# ============================================================================
# CLOSING
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""---

# 🎉 Selamat!

Anda telah menyelesaikan **20 latihan Python**!

## 📊 Progress Check:
- 🟢 Easy (1-7): Menguji pemahaman dasar
- 🟡 Medium (8-15): Menggabungkan beberapa konsep
- 🔴 Hard (16-20): Problem solving kompleks

## 🚀 Next Steps:

1. **Review Solutions** (`00_solutions.ipynb`)
   - Bandingkan solusi Anda dengan solusi yang disediakan
   - Pahami alternative approaches
   
2. **Practice More!**
   - HackerRank Python
   - LeetCode Easy problems
   - Codewars

3. **Move to Module 1**
   - Anda sudah siap untuk Data Science!
   
---

**Keep coding, keep learning!** 💪

<div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center;">
    <h2>💡 Remember</h2>
    <p><em>"Programming is learned by writing programs, not by reading about them."</em></p>
    <p>- Brian W. Kernighan</p>
</div>
"""))

# Save notebook
nb['cells'] = cells
with open('00_exercises.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✅ Exercises notebook created! Total: {len(cells)} cells")
print(f"📝 20 exercises covering all Python basics")
