import re 

text = """
1 
 문서번호  FG–1601 
제정일자 - 
개정일자  2022.09.01 
 
 
 
 
 
 
 
 
 
 
취 업 규 칙 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
"""
print(text, end='\n\n\n')
# text = re.sub(r" +", "", text)
text = re.sub(r'(\n\s*)+\n+', '\n', text)
text = re.sub("\·{1,}", " ", text)
text = re.sub("\.{1,}", ".", text)
print('after cleansing: ' + text)