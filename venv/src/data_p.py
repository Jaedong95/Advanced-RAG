class DataProcessor():
    def __init__(self, args):
        self.args = args 
    
    def cleanse_text(self, text):
        '''
        다중 줄바꿈 제거 및 특수 문자 중복 제거
        '''
        import re 
        text = re.sub(r'(\n\s*)+\n+', '\n\n', text)
        text = re.sub(r"\·{1,}", " ", text)
        text = re.sub(r"\.{1,}", ".", text)
        # print('after cleansing: ' + text)
        return text