import re

# exracting urls
def extract_url(text5):
    try:
     url = re.search("(?P<url>https?://[^\s]+)", text5).group("url")
    except:
        url = None
    return url

# extract address 
def extract_address(text):
    regexp = "[0-9]{1,3} .+, .+, [A-Z]{2} [0-9]{5}"
    address = re.findall(regexp, text)
    #addresses = pyap.parse(text, country='INDIA')
    return address

#find pincode
def extract_pincode(text):
    pincode =  r"[^\d][^a-zA-Z\d](\d{6})[^a-zA-Z\d]"
    pattern = re.compile(pincode)
    result = pattern.findall(text)
    if len(result)==0:
        return ' '
    return result[0]