## 🧩 Część C – Analiza parametrów i wnioski

### 1. Zmiany centroidów i przypisań punktów przy różnych wartościach k
- Dla **k = 2** punkty grupują się w dwa wyraźne zbiory – jeden skupiony w okolicy małych wartości współrzędnych (lewa dolna część wykresu), a drugi w prawej górnej części.
- Dla **k = 3** algorytm rozdziela jedną z istniejących grup na dwa mniejsze podzbiory. Centroidy przesuwają się, by lepiej odzwierciedlić lokalne skupienia punktów.
- Wraz ze wzrostem liczby klastrów centroidy stają się bardziej "lokalne", a granice między klastrami ostrzejsze.

---

### 2. Rozmyta przynależność (FCM) vs twarde przypisanie (K-means)
- W **K-means** każdy punkt należy **tylko do jednego** klastra – przypisanie jest binarne (twarde).
- W **Fuzzy C-Means** (FCM) punkty mają **stopień przynależności** do każdego klastra, co pozwala lepiej oddać niejednoznaczność danych.
- Punkty położone „na granicy” między klastrami mają w FCM przynależności zbliżone do 0.5 / 0.5, podczas gdy w K-means muszą zostać przypisane arbitralnie do jednego klastra.
- Dzięki temu FCM jest bardziej realistyczny w modelowaniu danych, gdzie granice między grupami nie są ostre.

---

### 3. Wpływ punktów odstających (szum w danych)
- Po dodaniu punktu odstającego (np. **P10 = [18.5, 11.5]**) algorytm **K-means** reaguje silnie – centroid jednego klastra przesuwa się w stronę punktu odstającego.
- Może to spowodować, że pozostałe punkty zostaną niepoprawnie przypisane lub klaster będzie reprezentowany tylko przez punkt odstający.
- W **Fuzzy C-Means** przynależność punktu odstającego rozmywa się (np. 0.6 / 0.4), co ogranicza jego wpływ na centroidy pozostałych grup.
- Dzięki temu FCM lepiej radzi sobie z danymi zawierającymi szum i pojedyncze odległe obserwacje.

---

### 4. Elastyczność algorytmów wobec punktów odstających
- **Fuzzy C-Means** jest bardziej elastyczny, ponieważ nie wymusza jednoznacznego przypisania punktu do klastra.
- Daje możliwość częściowej przynależności do wielu klastrów, co ogranicza wpływ pojedynczych ekstremalnych obserwacji.
- **K-means** jest bardziej wrażliwy, ponieważ centroid opiera się na średniej, która silnie reaguje na wartości odstające.

---

### 5. Wpływ liczby klastrów na wyniki
- Dla małej liczby klastrów (`k=2`) algorytmy tworzą ogólne, szerokie grupy, które łączą punkty o różnej charakterystyce.
- Dla większej liczby klastrów (`k=3` i więcej) pojawia się dokładniejszy, ale bardziej szczegółowy podział — niekiedy zbyt drobny (tzw. **overfitting**).
- W praktyce optymalną liczbę klastrów warto dobierać empirycznie, np. metodą **łokcia** lub **silhouette score**.

---

### 💡 Podsumowanie
- **K-means** jest szybki, prosty i intuicyjny, ale mniej odporny na szum i wymaga z góry określenia liczby klastrów.
- **Fuzzy C-Means** jest bardziej elastyczny, pozwala na analizę niejednoznacznych przypadków i lepiej radzi sobie z danymi rozmytymi lub zawierającymi odstające punkty.
- W przypadku danych z wyraźnymi granicami – K-means daje wystarczająco dobre wyniki.
- W przypadku danych niejednoznacznych lub z szumem – Fuzzy C-Means pozwala uzyskać bardziej realistyczne i stabilne grupowanie.
