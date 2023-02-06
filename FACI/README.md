# FACI

## Параметры

* gammaGrid -- набор коэффициентов в смещении для значений alpha. Выступает аналогом learning_rate, поэтому имеет схожие пределы (0.01 - 0.1). При оставлении только маленьких значений, кривая Local Coverage становится все ближе и ближе к кривой для константной alpha (так как очень маленькое изменение идет). 
* eta -- задается формулой из статьи, работает хорошо. Задает мультипликативную константу в экспоненте при перевзвешивании всех "веток развития". Чем больше значение, тем меньше вес всем удаленным от оптимального значения на данный момент (beta_t). В ноутбуке при большом значении среднее и Бернули почти совпадают -- потому что вес только у одного параметра большой, у остальных очень маленький.
* sigma -- коэффициент сглаживания для весов, чем меньше -- тем больше сглаживается к среднему значению. Как будто, не сильно критичный параметр.