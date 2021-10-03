#library
import pandas as pd 
import numpy as np 
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import ArrayDictionary, StopWordRemoverFactory, StopWordRemover
from nltk.tokenize import word_tokenize
from nlp_id.lemmatizer import Lemmatizer
import ast
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
import inflect
import nltk


# def convertText(text:str):
#   # text = ast.literal_eval(text)
#   text = ' '.join(text)
#   text = ''.join(text.split())
#   return text

def createSlang(filename):
  df = pd.read_csv(filename, header=0)
  return dict(df.values)

def addSastrawiStopword(stopwordList:list):
  stop_factory = StopWordRemoverFactory().get_stop_words()
  stopword_list = stop_factory + stopwordList 
  dictionary = ArrayDictionary(stopword_list)
  stopping = StopWordRemover(dictionary)
  return stopping

nlp = Lemmatizer()
def callingFunction(text:str, slang_words=None, sastrawi_stopwords=None, stopword=None, lemmatize = False):
  texts = re.sub('[^\w\s]', ' ', str(text).lower().strip())
  texts = texts.split()
  if lemmatize == True:
    texts = [nlp.lemmatize(txt) for txt in texts]
  if slang_words is not None:
    for i, word in enumerate(texts):
      if word in slang_words.keys():
        texts[i] = slang_words[word]
  if sastrawi_stopwords is not None:
    texts = [sastrawi_stopwords.remove(txt) for txt in texts]
  if stopword is not None:
    texts = [word for word in texts if word not in stopword]
  return ' '.join(texts)
  


extra_stopwords = ['ayo', 'dan', 'yg', 'ini', 'dengan', 'to','ya', 'yes', 'sih'
                    'itu', 'dari', 'ada', 'dalam', 'is', 'di',
                    'ga', 'ya', 'ke', 'el', 'the', 'jg',
                    'bgtu','sdh','org','krn','msh','utk',
                    'dgn','ni','sm','udh','bnyk','jgn',
                    'dll','jd','tp','pd','tdk','ambe',
                    'guys','gini','for','bgt','ah',
                    'nggak','biar','deh','nih','klo',
                    'yg','gak','nya','kl','bikin','bhw',
                    'ya','aja','ga','ha','si','from',
                    'the','at','yak','have','to','ini',
                    'itu','dari','dan','ada','dalam',
                    'is','ke','el','eh','yuk','kuy',
                    'blm','kan','klo','tetap','pada',
                    'pda','kalo','sama','yang','tapi',
                    'kok','lagi','di','juga','mau',
                    'of','dah','yah','tuh','emang','pas',
                    'lg','emg','karna','be','ngga','nah',
                    'RT', 'replying', 'to', 'adanya', 'adapun', 'agak', 'agaknya',
                    'agar', 'akulah', 'amat', 'amatlah', 'andalah', 'antar', 'antara', 'antaranya',
                    'apalagi', 'apatah', 'artinya', 'asal', 'asalkan', 'atas', 'bagai', 'bagaikan',
                    'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya', 'bakal', 'bakalan', 'balik',
                    'bapak', 'baru', 'bawah', 'beberapa', 'begini', 'beginian', 'beginikah',
                    'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja', 'belakang',
                    'belakangan', 'berada', 'berapalah', 'berbagai', 'berikut', 'berikutnya',
                    'berkenaan', 'berlainan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula'
                    'bersama', 'bersama-sama', 'berturut', 'berturut-turut', 'berupa', 'besar', 'betul',
                    'biasa', 'biasanya', 'bila', 'buat', 'bukannya', 'bulan', 'bung', 'cuma',
                    'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang', 'dekat', 'demi', 
                    'demikian', 'dengan', 'depan', 'dia', 'dialah', 'diantara', 'diantaranya', 'dibuat',
                    'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan', 'diibaratkannya',
                    'diketahuinya', 'dikira', 'dilakukan', 'dilalui', 'dilihat', 'dimaksud',
                    'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'dimisalkan', 'diperbuat',
                    'diperbuatnya', 'dipergunakan', 'dipersoalkan', 'dipunyai', 'diri', 'dirinya', 
                    'disebut', 'disebutkan', 'disebutkannya', 'disini', 'disinilah', 'ditunjuk', 
                    'ditunjuki', 'dong', 'dulu', 'guna', 'gunakan', 'hal', 'hanya', 'hanyalah', 
                    'hingga', 'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ini', 
                    'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah', 'jadinya', 
                    'jauh', 'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 
                    'kan', 'karenanya', 'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 
                    'keinginan', 'kelamaan', 'kelihatan', 'kelihatannya', 'kelima', 'keluar', 
                    'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya', 'kepada', 'kepadanya', 
                    'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah', 'kira', 'klo', 'kok', 
                    'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu', 'lanjut', 'lanjutnya', 
                    'lebih', 'lewat', 'luar', 'macam', 'maka', 'makanya', 'makin', 'malah', 
                    'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa', 
                    'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing', 
                    'mau', 'maupun', 'melainkan', 'melalui', 'melihat', 'melihatnya', 'memang', 
                    'membuat', 'memerlukan', 'memintakan', 'memisalkan', 'memperbuat', 
                    'mempergunakan', 'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 
                    'menambahkan', 'menanti', 'menanti-nanti', 'menantikan', 'mendatang', 
                    'mengenai', 'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 
                    'mengibaratkan', 'mengibaratkannya', 'menginginkan', 'mengira', 'menjadi', 
                    'menuju', 'menunjuk', 'menunjuki', 'menunjukan', 'menunjuknya', 'menyeluruh', 
                    'menyiapkan', 'merasa', 'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 
                    'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'mungkin', 'mungkinkah', 'nah', 
                    'naik', 'namun', 'nanti', 'nantinya', 'nyatanya', 'oleh', 'olehnya', 'pada', 
                    'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti', 
                    'pastilah', 'penting', 'pentingnya', 'per', 'pertama', 'pertama-tama', 'pihak', 
                    'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa', 'rasanya', 'rata', 'rupanya',
                    'saja', 'sajalah', 'saling', 'sambil', 'sampai', 'sampai-sampai', 'sana', 'sangat', 
                    'sangatlah', 'sayalah', 'se', 'sebab', 'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 
                    'sebagian', 'sebegini', 'sebegitu', 'sebelum', 'sebesar', 'sebisanya', 'sebuah', 'sebut', 
                    'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 
                    'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'sehingga', 'sejak',
                    'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 
                    'sekalian', 'sekaligus', 'sekalipun', 'sekarang', 'sekecil', 'seketika', 'sekiranya', 
                    'sekitar', 'sekitarnya', 'sekurang-kurangnya', 'sekurangnya', 'sela', 'selain', 'selaku', 
                    'selalu', 'selama', 'selama-lamanya', 'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 
                    'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 
                    'sementara', 'semisal', 'semisalnya', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian', 
                    'sendirinya', 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 
                    'seperti', 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta', 'seru', 'serupa', 'sesaat', 
                    'sesama', 'sesampai', 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 
                    'setelah', 'setempat', 'setengah', 'seterusnya', 'setiap', 'setiba', 'setibanya', 'setinggi', 
                    'seusai', 'sewaktu', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'supaya', 'tadi', 
                    'tadinya', 'tahu', 'tahun', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas', 
                    'tandasnya', 'tanpa', 'tapi', 'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah', 
                    'tentunya', 'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap', 
                    'terhadapnya', 'terjadi', 'terjadilah', 'terjadinya', 'terkira', 'terlalu', 'terlebih', 
                    'terlihat', 'termasuk', 'ternyata', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju', 
                    'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tiba-tiba', 'tinggi', 'toh', 
                    'tunjuk', 'turut', 'umum', 'umumnya', 'untuk', 'usah', 'usai', 'waduh', 'wah', 'wahai', 
                    'waktu', 'walau', 'walaupun', 'wong', 'yah', 'yaitu', 'yakin', 'yakni', 'yang', 'ada', 
                    'adalah', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'aku', 'anda', 'apa', 'apaan', 
                    'apabila', 'apakah', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagaimana', 
                    'bagaimanakah', 'bagaimanapun', 'baik', 'banyak', 'beginilah', 'belum', 'belumlah', 
                    'benar', 'benarkah', 'benarlah', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 
                    'berapakah', 'berapapun', 'berarti', 'berawal', 'berdatangan', 'beri', 'berikan', 
                    'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berlalu', 
                    'berlangsung', 'berlebihan', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 
                    'bertutur', 'berujar', 'betulkah', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 
                    'bolehlah', 'bukan', 'bukankah', 'bukanlah', 'cara', 'caranya', 'cukup', 'cukupkah', 
                    'cukuplah', 'demikianlah', 'di', 'diakhiri', 'diakhirinya', 'diberi', 'diberikan', 
                    'diberikannya', 'datangkan', 'diingat', 'diingatkan', 'diinginkan', 'dijawab', 
                    'dijelaskan', 'dijelaskannya', 'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan', 
                    'diketahui', 'diminta', 'dimintai', 'dimulai', 'dimulailah', 'dimulainya', 'dimungkinkan', 
                    'dini', 'dipastikan', 'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 
                    'dipertanyakan', 'disampaikan', 'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai', 
                    'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 
                    'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya', 'diungkapkan', 'dua', 
                    'empat', 'enggak', 'enggaknya', 'entah', 'entahlah', 'hampir', 'hari', 'harus', 'haruslah', 
                    'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'ikut', 'ingat', 'ingat-ingat', 'ingin',
                    'inginkah', 'inginkan', 'jangan', 'jangankan', 'janganlah', 'jawab', 'jawaban', 'jawabnya', 
                    'jelas', 'jelaskan', 'jelaslah', 'jelasnya', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 
                    'kami', 'kamilah', 'kamu', 'kamulah', 'kapan', 'kapankah', 'kapanpun', 'karena', 'kasus', 
                    'kata', 'katakan', 'katakanlah', 'katanya', 'kenapa', 'kesampaian', 'keseluruhan', 
                    'keseluruhannya', 'kira-kira', 'kiranya', 'kita', 'kitalah', 'kurang', 'lama', 'lamanya', 
                    'lima', 'melakukan', 'memastikan', 'memberi', 'memberikan', 'memihak', 'meminta', 
                    'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan', 
                    'menandaskan', 'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 
                    'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 
                    'mengingat', 'mengingatkan', 'mengucapkan', 'mengucapkannya', 'mengungkapkan', 
                    'menjawab', 'menjelaskan', 'menunjukkan', 'menurut', 'menuturkan', 'menyampaikan',
                    'menyangkut', 'menyatakan', 'menyebutkan', 'meyakini', 'meyakinkan', 'mula', 
                    'mulai', 'mulailah', 'mulanya', 'nyaris', 'percuma', 'perlu', 'perlukah', 'perlunya', 
                    'pernah', 'persoalan', 'pertanyaan', 'pertanyakan', 'saat', 'saatnya', 'sama', 'sama-sama', 
                  'sampaikan', 'satu', 'saya', 'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 
                    'sebanyak', 'sebelumnya', 'sebenarnya', 'seberapa', 'sebetulnya', 'segera', 'seharusnya', 
                    'seingat', 'sema', 'sa', 'sih', 'semaunya', 'sempat', 'seperlunya', 'sesegera', 
                    'setidak-tidaknya', 'setidaknya', 'siap', 'siapa', 'siapakah', 'siapapun', 'sudah', 
                    'sudahkah', 'sudahlah', 'ta', 'tanya', 'tanyakan', 'tanyanya', 'tegas', 'tegasnya', 
                    'tepat', 'terakhir', 'teringat', 'teringat-ingat', 'tersampaikan', 'tidak', 'tidakkah', 
                    'tidaklah', 'tiga', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 
                    'ujar', 'ujarnya', 'ungkap', 'ungkapnya', 'waktunya']

more_stopwords = ['belas', 'ratus', 'puluh', 'ribu', 'juta', 'satu', 'dua', 'tiga',
                  'empat','lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'nol',
                  'kayak', 'tidak', 'hai', 'me', 'done', 'ki', 'moga', 'bang',
                  'amin', 'kak', 'rplima', 'rpempat', 'rptiga', 'rpdua', 'rpsatu',
                  'rpenam', 'rptujuh', 'rpdelapan', 'rpsembilan', 'and', 'je', 
                  'nak', 'you', 'thank you']