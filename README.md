CryptoTax
=========

Do you need to file your taxes on crypto gains? Spoiler alert: It sucks.

This is an attempt to fill those tax forms for you automatically (or at least help you considerably) from exchange-exported trading data.

**NOTE:** This was primarily developed for use to file taxes in Sweden, which might lead to differences in how certain things are calculated or approached.

## WILL CONTAIN BUGS, ALWAYS CHECK THE END RESULT MANUALLY


# Usage

Install dependencies using poetry: `poetry install`

Some commands require price history to function, you can download it by running: `make get_data`

Run in the poetry-managed virtualenv: `poetry run cryptotax`

# Resources

 - [Kryptovalutor](https://www.skatteverket.se/privat/skatter/vardepapper/andratillgangar/kryptovalutor.4.15532c7b1442f256bae11b60.html) (Skatteverket)
 - [Så beskattas kryptovalutor](https://www.kryptovalutor.se/sa-beskattas-kryptovalutor/) (kryptovalutor.se)
