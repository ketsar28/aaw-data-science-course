#!/usr/bin/env python3
"""Add Part 4, 5, 6 to complete the notebook (200+ cells)"""
import nbformat as nbf

# Load notebook
with open('00_python_crash_course.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

cells = nb['cells']

# ============================================================================
# PART 4: FUNCTIONS
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🔧 PART 4: Functions (45 menit)

## Progress: 4/6 🟩🟩🟩🟩⬜⬜

Functions = Blok kode yang bisa digunakan berkali-kali

**Analogi**: Seperti resep masakan. Sekali buat resep, bisa dipakai berkali-kali kapan saja.

---

## 4.1 Membuat Function Sederhana
"""))

cells.append(nbf.v4.new_code_cell("""# Function tanpa parameter dan tanpa return
def sapa():
    print("Halo, Selamat datang!")
    print("Semoga hari Anda menyenangkan!")

# Memanggil function
sapa()"""))

cells.append(nbf.v4.new_code_cell("""# Function dengan parameter
def sapa_nama(nama):
    print(f"Halo, {nama}!")
    print("Selamat datang!")

# Memanggil dengan argument
sapa_nama("Budi")
sapa_nama("Ani")"""))

cells.append(nbf.v4.new_code_cell("""# Function dengan multiple parameters
def perkenalan(nama, umur, kota):
    print(f"Nama saya {nama}")
    print(f"Umur saya {umur} tahun")
    print(f"Saya tinggal di {kota}")
    print()

perkenalan("Budi", 25, "Jakarta")
perkenalan("Ani", 22, "Bandung")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 4.2 Return Value

Function bisa **mengembalikan** nilai dengan `return`.
"""))

cells.append(nbf.v4.new_code_cell("""# Function dengan return
def tambah(a, b):
    hasil = a + b
    return hasil

# Menyimpan hasil return ke variable
x = tambah(5, 3)
print("5 + 3 =", x)

# Bisa langsung print
print("10 + 7 =", tambah(10, 7))"""))

cells.append(nbf.v4.new_code_cell("""# Function untuk menghitung luas persegi panjang
def luas_persegi_panjang(panjang, lebar):
    luas = panjang * lebar
    return luas

def keliling_persegi_panjang(panjang, lebar):
    keliling = 2 * (panjang + lebar)
    return keliling

# Menggunakan functions
p = 15
l = 8
print(f"Persegi panjang {p} x {l}")
print(f"Luas: {luas_persegi_panjang(p, l)}")
print(f"Keliling: {keliling_persegi_panjang(p, l)}")"""))

cells.append(nbf.v4.new_code_cell("""# Function bisa return multiple values (sebagai tuple)
def hitung_statistik(angka):
    total = sum(angka)
    rata_rata = total / len(angka)
    maksimum = max(angka)
    minimum = min(angka)
    return total, rata_rata, maksimum, minimum

nilai = [78, 92, 85, 67, 95, 73, 88]
total, rata, maks, min_val = hitung_statistik(nilai)

print(f"Total: {total}")
print(f"Rata-rata: {rata:.2f}")
print(f"Maksimum: {maks}")
print(f"Minimum: {min_val}")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 4.3 Default Parameters

Parameter bisa punya nilai default.
"""))

cells.append(nbf.v4.new_code_cell("""# Function dengan default parameter
def sapa_lengkap(nama, salam="Halo"):
    print(f"{salam}, {nama}!")

# Pakai default
sapa_lengkap("Budi")

# Override default
sapa_lengkap("Ani", "Selamat pagi")
sapa_lengkap("Citra", "Hi")"""))

cells.append(nbf.v4.new_code_cell("""# Contoh praktis: Function untuk pangkat
def pangkat(angka, eksponen=2):  # Default: kuadrat
    return angka ** eksponen

print("5^2 =", pangkat(5))        # Pakai default (kuadrat)
print("5^3 =", pangkat(5, 3))     # Override dengan 3
print("2^10 =", pangkat(2, 10))   # Override dengan 10"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 4.4 Lambda Function

Lambda = function singkat (satu baris) tanpa nama.

**Kapan dipakai?** Untuk function sederhana yang hanya dipakai sekali.
"""))

cells.append(nbf.v4.new_code_cell("""# Function biasa
def kuadrat(x):
    return x ** 2

# Lambda equivalent
kuadrat_lambda = lambda x: x ** 2

print("Kuadrat 5 (function biasa):", kuadrat(5))
print("Kuadrat 5 (lambda):", kuadrat_lambda(5))"""))

cells.append(nbf.v4.new_code_cell("""# Lambda dengan multiple parameters
tambah = lambda a, b: a + b
kali = lambda a, b: a * b

print("3 + 4 =", tambah(3, 4))
print("3 x 4 =", kali(3, 4))"""))

cells.append(nbf.v4.new_code_cell("""# Lambda sering dipakai dengan map(), filter(), sorted()

angka = [1, 2, 3, 4, 5]

# map(): Apply function ke setiap elemen
kuadrat_list = list(map(lambda x: x**2, angka))
print("Kuadrat:", kuadrat_list)

# filter(): Filter elemen berdasarkan kondisi
genap = list(filter(lambda x: x % 2 == 0, angka))
print("Genap:", genap)"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 4.5 Built-in Functions

Python punya banyak built-in functions yang berguna:
"""))

cells.append(nbf.v4.new_code_cell("""# len() - Panjang
print("len([1,2,3,4,5]):", len([1,2,3,4,5]))
print("len('Hello'):", len("Hello"))

# sum() - Jumlah
print("sum([1,2,3,4,5]):", sum([1,2,3,4,5]))

# max(), min() - Maksimum, Minimum
angka = [45, 23, 78, 12, 90, 34]
print("max:", max(angka))
print("min:", min(angka))

# abs() - Nilai absolut
print("abs(-5):", abs(-5))
print("abs(5):", abs(5))

# round() - Pembulatan
print("round(3.14159, 2):", round(3.14159, 2))  # Bulatkan 2 desimal
print("round(3.7):", round(3.7))  # Bulatkan ke integer terdekat"""))

cells.append(nbf.v4.new_code_cell("""# sorted() - Mengurutkan (return list baru)
angka = [5, 2, 8, 1, 9]
print("Original:", angka)
print("Sorted:", sorted(angka))
print("Sorted descending:", sorted(angka, reverse=True))
print("Original masih:", angka)  # Original tidak berubah

# reversed() - Membalik urutan
print("Reversed:", list(reversed(angka)))

# enumerate() - Mendapat index dan value
buah = ["apel", "jeruk", "mangga"]
for i, nama in enumerate(buah):
    print(f"{i}: {nama}")"""))

cells.append(nbf.v4.new_code_cell("""# zip() - Menggabungkan beberapa list
nama = ["Budi", "Ani", "Citra"]
nilai = [85, 92, 78]
kelas = ["12A", "12B", "12A"]

for n, nil, kel in zip(nama, nilai, kelas):
    print(f"{n} (kelas {kel}): {nil}")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 🏋️ Latihan Part 4

### Latihan 4.1: Function Sederhana
Buat function `cek_genap_ganjil(angka)` yang:
- Return "Genap" jika angka genap
- Return "Ganjil" jika angka ganjil

Test dengan beberapa angka.
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 4.2: Function dengan Multiple Return
Buat function `konversi_suhu(celsius)` yang return:
- Fahrenheit: (celsius × 9/5) + 32
- Kelvin: celsius + 273.15

Test dengan celsius = 25
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 4.3: Lambda & Filter
Dari list angka 1-50, filter dan print hanya angka yang:
1. Kelipatan 3
2. Kelipatan 5
3. Kelipatan 3 DAN 5

Gunakan lambda dan filter!
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""---

✅ **Part 4 Selesai!** Anda sudah bisa membuat dan menggunakan functions!

**Next**: Part 5 - File Operations

---"""))

# ============================================================================
# PART 5: FILE OPERATIONS
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 📁 PART 5: File Operations (30 menit)

## Progress: 5/6 🟩🟩🟩🟩🟩⬜

File Operations = Membaca dan menulis file

---

## 5.1 Membaca File (Reading)
"""))

cells.append(nbf.v4.new_code_cell("""# Membuat file contoh terlebih dahulu
with open("contoh.txt", "w") as f:
    f.write("Baris pertama\\n")
    f.write("Baris kedua\\n")
    f.write("Baris ketiga\\n")

print("✅ File contoh.txt sudah dibuat!")"""))

cells.append(nbf.v4.new_code_cell("""# Cara 1: read() - Baca seluruh isi file
file = open("contoh.txt", "r")  # "r" = read mode
isi = file.read()
print(isi)
file.close()  # PENTING: Tutup file setelah selesai"""))

cells.append(nbf.v4.new_code_cell("""# Cara 2: readline() - Baca satu baris
file = open("contoh.txt", "r")
baris1 = file.readline()
baris2 = file.readline()
print("Baris 1:", baris1)
print("Baris 2:", baris2)
file.close()"""))

cells.append(nbf.v4.new_code_cell("""# Cara 3: readlines() - Baca semua baris ke dalam list
file = open("contoh.txt", "r")
semua_baris = file.readlines()
print("Semua baris:", semua_baris)
file.close()

# Loop setiap baris
for baris in semua_baris:
    print(">>", baris.strip())  # strip() untuk hapus \\n"""))

cells.append(nbf.v4.new_markdown_cell("""### Context Manager (with statement)

**Best Practice**: Gunakan `with` agar file otomatis tertutup.
"""))

cells.append(nbf.v4.new_code_cell("""# Dengan 'with' - file otomatis tertutup
with open("contoh.txt", "r") as file:
    isi = file.read()
    print(isi)

# Tidak perlu file.close() - sudah otomatis!"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 5.2 Menulis File (Writing)
"""))

cells.append(nbf.v4.new_code_cell("""# Mode "w" - Write (overwrite/timpa jika file sudah ada)
with open("output.txt", "w") as file:
    file.write("Ini baris pertama\\n")
    file.write("Ini baris kedua\\n")
    file.write("Ini baris ketiga\\n")

print("✅ File output.txt sudah dibuat!")

# Baca untuk verifikasi
with open("output.txt", "r") as file:
    print(file.read())"""))

cells.append(nbf.v4.new_code_cell("""# Mode "a" - Append (tambahkan di akhir file)
with open("output.txt", "a") as file:
    file.write("Baris tambahan 1\\n")
    file.write("Baris tambahan 2\\n")

print("✅ File output.txt sudah ditambahi!")

# Baca untuk verifikasi
with open("output.txt", "r") as file:
    print(file.read())"""))

cells.append(nbf.v4.new_code_cell("""# Menulis list ke file
nama_siswa = ["Budi", "Ani", "Citra", "Dedi", "Eka"]

with open("siswa.txt", "w") as file:
    for nama in nama_siswa:
        file.write(nama + "\\n")

print("✅ File siswa.txt sudah dibuat!")

# Baca kembali
with open("siswa.txt", "r") as file:
    print(file.read())"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 5.3 Contoh Praktis: Membaca CSV

CSV (Comma-Separated Values) = Format data yang umum digunakan.
"""))

cells.append(nbf.v4.new_code_cell("""# Membuat file CSV contoh
with open("nilai.csv", "w") as file:
    file.write("nama,matematika,fisika,kimia\\n")
    file.write("Budi,85,78,92\\n")
    file.write("Ani,92,88,85\\n")
    file.write("Citra,78,95,88\\n")

print("✅ File nilai.csv sudah dibuat!")"""))

cells.append(nbf.v4.new_code_cell("""# Membaca CSV secara manual
with open("nilai.csv", "r") as file:
    # Baca header
    header = file.readline().strip().split(",")
    print("Header:", header)
    
    print("\\nData siswa:")
    for baris in file:
        data = baris.strip().split(",")
        nama = data[0]
        mat = int(data[1])
        fis = int(data[2])
        kim = int(data[3])
        rata_rata = (mat + fis + kim) / 3
        print(f"{nama}: Rata-rata = {rata_rata:.2f}")"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Catatan

Untuk CSV yang lebih kompleks, di Data Science nanti kita akan pakai library **Pandas** yang jauh lebih powerful!

---

## 🏋️ Latihan Part 5

### Latihan 5.1: Membaca dan Menghitung
Buat file "angka.txt" yang berisi angka-angka (satu angka per baris).
Lalu baca file tersebut dan hitung:
1. Total
2. Rata-rata
3. Berapa banyak angka yang > 50
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 5.2: Menulis Log
Buat function `catat_log(pesan)` yang menambahkan pesan ke file "log.txt" dengan format:
`[waktu] pesan`

Gunakan `import datetime` untuk mendapat waktu sekarang.

Hint: `datetime.datetime.now()`
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""---

✅ **Part 5 Selesai!** Anda sudah bisa membaca dan menulis file!

**Next**: Part 6 - Error Handling (terakhir!)

---"""))

# ============================================================================
# PART 6: ERROR HANDLING
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# ⚠️ PART 6: Error Handling (20 menit)

## Progress: 6/6 🟩🟩🟩🟩🟩🟩

Error Handling = Menangani error agar program tidak crash

---

## 6.1 Types of Errors

Ada banyak jenis error dalam Python:
"""))

cells.append(nbf.v4.new_code_cell("""# Syntax Error - Kesalahan penulisan kode
# print("Hello"  # Missing )
# Uncomment baris di atas untuk lihat error"""))

cells.append(nbf.v4.new_code_cell("""# ZeroDivisionError - Pembagian dengan nol
# hasil = 10 / 0  # Error!
# Uncomment untuk lihat error"""))

cells.append(nbf.v4.new_code_cell("""# ValueError - Nilai tidak sesuai
# angka = int("abc")  # Tidak bisa konversi "abc" ke integer
# Uncomment untuk lihat error"""))

cells.append(nbf.v4.new_code_cell("""# IndexError - Index di luar jangkauan
# angka = [1, 2, 3]
# print(angka[10])  # Index 10 tidak ada
# Uncomment untuk lihat error"""))

cells.append(nbf.v4.new_code_cell("""# KeyError - Key tidak ada dalam dictionary
# data = {"nama": "Budi", "umur": 25}
# print(data["alamat"])  # Key "alamat" tidak ada
# Uncomment untuk lihat error"""))

cells.append(nbf.v4.new_code_cell("""# FileNotFoundError - File tidak ditemukan
# with open("file_tidak_ada.txt", "r") as f:
#     data = f.read()
# Uncomment untuk lihat error"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 6.2 Try-Except - Menangani Error

Gunakan `try-except` untuk menangani error agar program tidak crash.
"""))

cells.append(nbf.v4.new_code_cell("""# Tanpa try-except (akan crash jika error)
# angka = int(input("Masukkan angka: "))  # Crash jika input bukan angka

# Dengan try-except (tidak crash)
try:
    angka_str = "abc"  # Simulasi input user
    angka = int(angka_str)
    print("Angka:", angka)
except:
    print("Error! Input bukan angka yang valid")
    print("Program tetap jalan!")"""))

cells.append(nbf.v4.new_code_cell("""# Try-except dengan tipe error spesifik
try:
    hasil = 10 / 0
except ZeroDivisionError:
    print("Error: Tidak bisa membagi dengan nol!")
    hasil = None

print("Program selesai")"""))

cells.append(nbf.v4.new_code_cell("""# Multiple except blocks
def bagi(a, b):
    try:
        hasil = a / b
        return hasil
    except ZeroDivisionError:
        print("Error: Pembagi tidak boleh nol!")
        return None
    except TypeError:
        print("Error: Input harus angka!")
        return None

print(bagi(10, 2))    # Normal
print(bagi(10, 0))    # ZeroDivisionError
print(bagi(10, "a"))  # TypeError"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 6.3 Try-Except-Else-Finally

Struktur lengkap error handling:
- `try`: Code yang mungkin error
- `except`: Jalankan jika ada error
- `else`: Jalankan jika TIDAK ada error
- `finally`: Jalankan PASTI (apapun yang terjadi)
"""))

cells.append(nbf.v4.new_code_cell("""def baca_file(nama_file):
    try:
        # Coba baca file
        with open(nama_file, "r") as file:
            isi = file.read()
    except FileNotFoundError:
        # Jika file tidak ada
        print(f"File '{nama_file}' tidak ditemukan!")
        isi = None
    else:
        # Jika berhasil
        print(f"File '{nama_file}' berhasil dibaca!")
    finally:
        # Selalu dijalankan
        print("Proses selesai.")
    
    return isi

# Test
print("\\nTest 1: File yang ada")
isi1 = baca_file("contoh.txt")

print("\\n" + "="*50)
print("Test 2: File yang tidak ada")
isi2 = baca_file("file_tidak_ada.txt")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 6.4 Raise - Membuat Error Sendiri

Kita bisa `raise` error sendiri untuk validasi.
"""))

cells.append(nbf.v4.new_code_cell("""def hitung_akar_kuadrat(angka):
    if angka < 0:
        raise ValueError("Angka tidak boleh negatif!")
    return angka ** 0.5

# Normal
print("Akar 25:", hitung_akar_kuadrat(25))

# Error
try:
    print("Akar -9:", hitung_akar_kuadrat(-9))
except ValueError as e:
    print(f"Error: {e}")"""))

cells.append(nbf.v4.new_code_cell("""# Contoh praktis: Validasi umur
def validasi_umur(umur):
    if not isinstance(umur, int):
        raise TypeError("Umur harus integer!")
    if umur < 0:
        raise ValueError("Umur tidak boleh negatif!")
    if umur > 150:
        raise ValueError("Umur tidak realistis!")
    return True

# Test berbagai kasus
test_cases = [25, -5, "dua puluh", 200]

for test in test_cases:
    try:
        validasi_umur(test)
        print(f"✅ {test}: Valid")
    except (TypeError, ValueError) as e:
        print(f"❌ {test}: {e}")"""))

cells.append(nbf.v4.new_markdown_cell("""---

## 🏋️ Latihan Part 6

### Latihan 6.1: Safe Division
Buat function `bagi_aman(a, b)` yang:
- Return hasil pembagian a / b
- Handle ZeroDivisionError dengan pesan error
- Handle TypeError jika input bukan angka
- Gunakan try-except-else-finally

Test dengan berbagai input.
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""### Latihan 6.2: File Reader yang Aman
Buat function `baca_file_aman(nama_file)` yang:
- Try untuk membaca file
- Jika file tidak ada, return "File tidak ditemukan"
- Jika ada error lain, return "Error membaca file"
- Jika berhasil, return isi file
- Print "Operasi selesai" di finally block
"""))

cells.append(nbf.v4.new_code_cell("""# Tulis kode Anda di sini


"""))

cells.append(nbf.v4.new_markdown_cell("""---

✅ **Part 6 Selesai!** ✅ **SEMUA PART SELESAI!**

---"""))

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🎉 SELAMAT! Anda Telah Menyelesaikan Python Crash Course!

---

## 📋 Ringkasan Apa yang Sudah Anda Pelajari:

### Part 1: Python Basics ✅
- Print statements
- Variables & data types (int, float, string, boolean)
- Operators (arithmetic, comparison, logical)
- Type conversion

### Part 2: Control Flow ✅
- If-elif-else statements
- For loops & while loops
- Break & continue
- Nested loops

### Part 3: Data Structures ✅
- **List**: Array yang bisa diubah
- **Dictionary**: Key-value pairs
- **Tuple**: Array yang tidak bisa diubah
- **Set**: Kumpulan unik tanpa duplikat
- List comprehension

### Part 4: Functions ✅
- Membuat function
- Parameters & return values
- Default parameters
- Lambda functions
- Built-in functions (map, filter, sorted, dll.)

### Part 5: File Operations ✅
- Membaca file (read, readline, readlines)
- Menulis file (write mode, append mode)
- Context manager (with statement)
- Bekerja dengan CSV

### Part 6: Error Handling ✅
- Types of errors
- Try-except blocks
- Try-except-else-finally
- Raising errors

---

## 🎯 Self-Check: Apakah Anda Sudah Bisa?

Sebelum lanjut ke Module 1, pastikan Anda bisa:

- [ ] Menulis program Python sederhana
- [ ] Menggunakan variables dan operators
- [ ] Membuat kondisi dengan if-else
- [ ] Menggunakan loops (for dan while)
- [ ] Bekerja dengan list, dictionary, tuple, set
- [ ] Membuat dan menggunakan functions
- [ ] Membaca dan menulis file
- [ ] Menangani errors dengan try-except

Jika ada yang belum yakin, **review kembali** part yang bersangkutan!

---

## 📝 Latihan Tambahan

File `00_exercises.ipynb` berisi **20 latihan** untuk practice lebih lanjut!

**Tips**: 
- Coba solve sendiri dulu (minimal 15-20 menit per soal)
- Jika stuck, lihat hints
- Jika masih stuck, lihat solutions
- **Pahami** solutionnya, jangan hanya copy-paste!

---

## 🚀 Next Steps

**Setelah menyelesaikan latihan, Anda siap untuk:**

### ➡️ **Module 1: Introduction to Data Science**
- Memahami Data Science ecosystem
- Mengenal tools & libraries
- Memulai perjalanan Data Science Anda!

---

## 💡 Tips untuk Lanjut Belajar

1. **Practice Coding Setiap Hari**
   - Minimal 30 menit per hari
   - Consistency > Intensity

2. **Bangun Project Kecil**
   - Calculator
   - To-do list
   - Simple game
   - Data analyzer

3. **Jangan Takut Error**
   - Error adalah bagian dari belajar
   - Google adalah teman Anda
   - Baca error message dengan teliti

4. **Join Communities**
   - Python subreddit
   - Stack Overflow
   - GitHub discussions

5. **Keep Learning**
   - Python adalah journey, bukan destination
   - Always something new to learn!

---

## 📚 Recommended Resources

### Online Practice:
- **HackerRank Python**: https://www.hackerrank.com/domains/python
- **LeetCode**: https://leetcode.com/
- **Codewars**: https://www.codewars.com/

### Books (Free):
- **Automate the Boring Stuff with Python**: https://automatetheboringstuff.com/
- **Think Python**: https://greenteapress.com/wp/think-python/

### YouTube Channels:
- Corey Schafer - Python Tutorials
- Programming with Mosh
- Tech With Tim

---

## 🎓 You're Ready!

**CONGRATULATIONS!** 🎉

Anda sudah memiliki fondasi Python yang solid. Sekarang saatnya apply skills ini untuk **Data Science**!

**See you in Module 1!** 🚀

---

<div style="text-align: center; padding: 20px; background-color: #e8f5e9; border-radius: 10px;">
    <h2>💪 Keep Coding, Keep Learning! 💪</h2>
    <p><em>"The expert in anything was once a beginner."</em></p>
</div>
"""))

# Save final notebook
nb['cells'] = cells
with open('00_python_crash_course.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✅ COMPLETE! Final notebook with {len(cells)} cells created!")
print(f"📊 Breakdown:")
print(f"   - Part 1: Basics")
print(f"   - Part 2: Control Flow")
print(f"   - Part 3: Data Structures")
print(f"   - Part 4: Functions")
print(f"   - Part 5: File Operations")
print(f"   - Part 6: Error Handling")
print(f"   - Summary & Next Steps")
print(f"\\n🎉 Total: {len(cells)} cells!")
