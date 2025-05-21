import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier # Będziemy symulować, ale dobrze mieć import

# --- Konfiguracja i dane stałe ---
DATA_FILE = 'data.csv'
PRODUCTS = {
    "Ultrasonografy przenośne": ["USG-Mobile-Pro", "USG-Portable-Lite", "USG-Clinic-Max"],
    "Głowice USG": ["Glowica-Liniowa-Slim", "Glowica-Konweksowa-Pro", "Glowica-Mikrokonweksowa-Vet"],
    "Platforma TU2": ["TU2-Standard", "TU2-Pro", "TU2-Enterprise"],
    "Moduły AI w TU2": ["AI-Module-Basic", "AI-Module-Advanced"],
    "Chmura danych (Cloud)": ["Cloud-50GB", "Cloud-200GB", "Cloud-Unlimited"],
    "Obsługa techniczna": ["Serwis-Standard", "Serwis-Premium"],
    "Szkolenia / edukacja": ["Webinar-Diagnostyka", "Kurs-AI-Wet", "Onboarding-Pro"],
    "Sprzęt towarzyszący": ["Etui-USG", "Uchwyt-Scannerski"]
}

# Symulowane dane do logowania
VALID_USERNAME = "handlowiec"
VALID_PASSWORD = "veteye_ai"

# --- Funkcje pomocnicze ---

@st.cache_data # Cache'owanie danych, aby aplikacja działała szybciej
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except FileNotFoundError:
        st.error(f"Plik danych '{DATA_FILE}' nie został znaleziony. Upewnij się, że jest w tym samym katalogu co aplikacja.")
        st.stop()

@st.cache_resource # Cache'owanie modeli, aby nie trenowały się przy każdej interakcji
def train_and_simulate_models(df_raw):
    # Kopia danych, aby nie modyfikować oryginalnego DataFrame
    df = df_raw.copy()

    # Przygotowanie danych (kodowanie kategorialnych)
    le = LabelEncoder()
    df['segment_encoded'] = le.fit_transform(df['segment'])
    # country celowo pomijamy w modelu zgodnie z ustaleniami
    
    features = [
        "clinic_size", "devices_owned", "last_purchase_days_ago", "purchase_count",
        "avg_purchase_value", "tu2_active", "tu2_sessions_last_30d", "ai_usage_ratio",
        "last_contact_days_ago", "open_rate", "click_rate", "support_tickets_last_6m",
        "segment_encoded"
    ]
    
    X = df[features]
    
    # Symulacja trenowania modeli (dla POC nie trenujemy realnie na 1000 rekordów)
    # W prawdziwym systemie byłoby to:
    # model_buy = XGBClassifier().fit(X, df['buy_label'])
    # model_churn = XGBClassifier().fit(X, df['churn_label'])
    
    # Na potrzeby POC i DEMONSTRATORA: symulujemy wyniki scoringu
    # Wytrenujmy bardzo proste modele, aby było coś realnego, ale szybko
    X_train, X_test, y_buy_train, y_buy_test, y_churn_train, y_churn_test = train_test_split(
        X, df['buy_label'], df['churn_label'], test_size=0.2, random_state=42
    )

    # Model Sprzedażowy
    model_buy = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_buy.fit(X_train, y_buy_train)
    df['upsell_propensity_score'] = model_buy.predict_proba(X)[:, 1] # Prawdopodobieństwo klasy 1

    # Model Antychurnowy
    model_churn = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_churn.fit(X_train, y_churn_train)
    df['churn_risk_score'] = model_churn.predict_proba(X)[:, 1] # Prawdopodobieństwo klasy 1

    # Dodaj symulowane SHAP values dla demo
    # W prawdziwym systemie użylibyśmy shap.Explainer i shap.values
    # Tutaj symulujemy, że niektóre cechy mają większy wpływ
    
    # Lista kluczowych cech dla SHAP (zgodnie z naszymi cechami)
    shap_features_map_buy = {
        'purchase_count': 0.2, 'avg_purchase_value': 0.15, 'tu2_active': 0.1, 
        'tu2_sessions_last_30d': 0.25, 'click_rate': 0.1, 'devices_owned': 0.2
    }
    shap_features_map_churn = {
        'tu2_sessions_last_30d': -0.3, 'last_contact_days_ago': -0.2, 'support_tickets_last_6m': -0.2,
        'tu2_active': -0.1, 'ai_usage_ratio': -0.1, 'last_purchase_days_ago': -0.1
    }

    return df, model_buy, model_churn, features, shap_features_map_buy, shap_features_map_churn

def get_product_recommendation(client_data):
    # Logika oparta na regułach eksperckich
    # Przykład: Jeśli klient ma tylko 1 urządzenie i wysoką aktywność TU2, ale niski AI usage ratio -> zaproponuj moduł AI lub szkolenie
    # Jeśli klient ma wiele urządzeń i wysoką wartość zakupu, ale dawno nie kupował -> zaproponuj nowe USG lub upgrade
    # Jeśli klient ma wysoki churn_risk_score -> zaproponuj serwis premium / wsparcie
    
    if client_data['churn_risk_score'] > 0.6: # Wysokie ryzyko churnu
        return random.choice(PRODUCTS["Obsługa techniczna"] + PRODUCTS["Szkolenia / edukacja"]) + ", aby poprawić doświadczenia."
    elif client_data['upsell_propensity_score'] > 0.7: # Wysoki potencjał upsell
        if client_data['devices_owned'] < 3 and client_data['tu2_active'] == 1 and client_data['ai_usage_ratio'] < 0.5:
            return random.choice(PRODUCTS["Moduły AI w TU2"]) + ", aby zwiększyć funkcjonalność."
        elif client_data['last_purchase_days_ago'] > 180 and client_data['avg_purchase_value'] > 10000:
            return random.choice(PRODUCTS["Ultrasonografy przenośne"]) + ", aby odświeżyć sprzęt."
        elif client_data['devices_owned'] >=3 and client_data['tu2_active'] == 1 and client_data['tu2_sessions_last_30d'] > 30:
            return random.choice(PRODUCTS["Głowice USG"]) + ", aby rozszerzyć diagnostykę."
        else: # Ogólna rekomendacja dla wysokiego potencjału
            return random.choice(PRODUCTS["Chmura danych (Cloud)"] + PRODUCTS["Platforma TU2"]) + ", aby usprawnić przepływ pracy."
    else: # Niska ocena, ogólna propozycja
        return "Brak konkretnej rekomendacji, sugerowane utrzymanie relacji."

def get_contact_strategy(client_data):
    if client_data['churn_risk_score'] > 0.6:
        return "PILNY TELEFON - dział Retencji / Customer Success."
    elif client_data['upsell_propensity_score'] > 0.7:
        return "TELEFON + E-MAIL z ofertą DEMO."
    elif client_data['upsell_propensity_score'] > 0.5:
        return "E-MAIL z personalizowaną ofertą."
    else:
        return "Standardowy kontakt (np. newsletter, przypomnienie)."

def get_conversation_script(client_data, recommendation):
    if "PILNY TELEFON" in get_contact_strategy(client_data):
        return f"""
        **Scenariusz rozmowy (Retencja):**
        "Dzień dobry, dzwonię z Vet-Eye S.A. w sprawie Państwa doświadczeń z naszym sprzętem/usługami. Zauważyliśmy, że Państwa aktywność nieco spadła/mogą Państwo potrzebować wsparcia w związku z {recommendation}. Chcielibyśmy upewnić się, że wszystko jest w porządku i zaoferować pomoc, aby w pełni wykorzystać potencjał Vet-Eye. Czy jest coś, w czym możemy Państwu pomóc?"
        """
    elif client_data['upsell_propensity_score'] > 0.7:
        return f"""
        **Scenariusz rozmowy (Sprzedaż/Upsell):**
        "Dzień dobry, dzwonię z Vet-Eye S.A. Z naszej analizy wynika, że Państwa klinika/praktyka może znacząco zyskać na wdrożeniu {recommendation}. Wielu naszych klientów w Państwa segmencie z sukcesem wdrożyło to rozwiązanie, poprawiając... [wymień korzyści]. Czy byliby Państwo zainteresowani krótką prezentacją online, abyśmy mogli omówić szczegóły?"
        """
    else:
        return "Brak specyficznego skryptu. Standardowe zapytanie o zadowolenie i potrzeby."

# --- Funkcja do symulacji SHAP ---
def plot_simulated_shap(features, client_data, shap_map, title="Wpływ cech na wynik"):
    # Tworzymy słownik z wartościami SHAP dla konkretnego klienta
    # Dla POC, bierzemy stałe wagi z mapy i mnożymy przez wartość cechy (lub jej binarną reprezentację)
    shap_values_dict = {}
    for feature, base_weight in shap_map.items():
        if feature in client_data:
            value = client_data[feature]
            # Prosta symulacja: jeśli cecha jest aktywna/duża, jej wpływ jest silniejszy
            if 'active' in feature or 'sessions' in feature or 'ratio' in feature or 'count' in feature or 'value' in feature:
                shap_value = base_weight * (value / np.mean(client_data[feature] if isinstance(client_data, pd.DataFrame) else [client_data[f] for f in features])) # Scale by average
            elif 'contact' in feature or 'purchase_days_ago' in feature or 'tickets' in feature: # Im większa, tym gorsza dla churn
                shap_value = base_weight * (value / np.mean(client_data[feature] if isinstance(client_data, pd.DataFrame) else [client_data[f] for f in features]))
            else: # binarna lub inna
                shap_value = base_weight * value
            shap_values_dict[feature] = shap_value
        else: # Handle cases where feature might not be directly in client_data (e.g. encoded)
             shap_values_dict[feature] = 0.0 # No impact

    # Sortowanie dla lepszej wizualizacji (największy wpływ na górze)
    sorted_shap_values = sorted(shap_values_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    
    # Wybieramy tylko te, które mają niezerowy wpływ
    filtered_shap_values = [(k, v) for k, v in sorted_shap_values if abs(v) > 0.01]

    # Stworzenie DataFrame dla Plotly
    if not filtered_shap_values:
        st.write("Brak znaczących wpływów cech do wizualizacji dla tego klienta.")
        return

    shap_df = pd.DataFrame(filtered_shap_values, columns=['Cecha', 'Wpływ SHAP (symulowany)'])
    
    fig = px.bar(shap_df, x='Wpływ SHAP (symulowany)', y='Cecha', orientation='h',
                 title=title,
                 color='Wpływ SHAP (symulowany)',
                 color_continuous_scale=px.colors.diverging.RdYlGn, # Czerwony dla negatywnego, zielony dla pozytywnego
                 labels={'Wpływ SHAP (symulowany)': 'Wpływ na wynik modelu'},
                 height=max(400, len(filtered_shap_values) * 50))
    fig.update_layout(showlegend=False, xaxis_title="Wpływ", yaxis_title="Cecha")
    st.plotly_chart(fig, use_container_width=True)


# --- Główna aplikacja Streamlit ---

st.set_page_config(
    page_title="Vet-Eye S.A. | Inteligentny CRM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sesja logowania
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("Logowanie do Inteligentnego CRM Vet-Eye S.A.")
    with st.form("login_form"):
        username = st.text_input("Nazwa użytkownika")
        password = st.text_input("Hasło", type="password")
        submit_button = st.form_submit_button("Zaloguj")

        if submit_button:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state['logged_in'] = True
                st.success("Zalogowano pomyślnie!")
                st.experimental_rerun() # Odśwież stronę po zalogowaniu
            else:
                st.error("Nieprawidłowa nazwa użytkownika lub hasło.")
    st.stop() # Zatrzymuje wyświetlanie reszty aplikacji, jeśli nie zalogowano

# Jeśli zalogowano, kontynuuj
st.sidebar.title("Vet-Eye S.A.")
st.sidebar.header("Inteligentny CRM")

menu_options = {
    "Dashboard": "🏠 Dashboard",
    "Klienci & Scoring": "👥 Klienci & Scoring",
    "Analiza Modeli AI": "🧠 Analiza Modeli AI",
    "Opis Projektu & POC": "📝 Opis Projektu & POC"
}
selected_option = st.sidebar.radio("Nawigacja", list(menu_options.keys()), format_func=lambda x: menu_options[x])

# Wyloguj
if st.sidebar.button("Wyloguj"):
    st.session_state['logged_in'] = False
    st.experimental_rerun()


# --- Ładowanie i przetwarzanie danych ---
df_raw = load_data()
df, model_buy, model_churn, features_for_models, shap_map_buy, shap_map_churn = train_and_simulate_models(df_raw.copy())


# --- Sekcja: Dashboard ---
if selected_option == "Dashboard":
    st.header("Dashboard Przeglądowy")
    st.write("Szybki wgląd w kluczowe wskaźniki systemu Inteligentnego CRM.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Liczba Klientów", len(df))
    with col2:
        st.metric("Średni Potencjał Sprzedażowy", f"{df['upsell_propensity_score'].mean():.2f}")
    with col3:
        st.metric("Średnie Ryzyko Churnu", f"{df['churn_risk_score'].mean():.2f}")

    st.markdown("---")

    st.subheader("Top 10 Klientów z Najwyższym Potencjałem Sprzedażowym")
    top_upsell = df.sort_values(by='upsell_propensity_score', ascending=False).head(10)
    st.dataframe(top_upsell[['client_id', 'segment', 'country', 'upsell_propensity_score', 'churn_risk_score']].style.format({
        'upsell_propensity_score': "{:.2f}", 'churn_risk_score': "{:.2f}"
    }), hide_index=True)

    st.subheader("Top 10 Klientów z Najwyższym Ryzykiem Churnu")
    top_churn = df.sort_values(by='churn_risk_score', ascending=False).head(10)
    st.dataframe(top_churn[['client_id', 'segment', 'country', 'upsell_propensity_score', 'churn_risk_score']].style.format({
        'upsell_propensity_score': "{:.2f}", 'churn_risk_score': "{:.2f}"
    }), hide_index=True)

    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_upsell_dist = px.histogram(df, x='upsell_propensity_score', nbins=20, 
                                       title='Rozkład Potencjału Sprzedażowego',
                                       labels={'upsell_propensity_score': 'Potencjał Sprzedażowy (0-1)'})
        st.plotly_chart(fig_upsell_dist, use_container_width=True)
    with col_chart2:
        fig_churn_dist = px.histogram(df, x='churn_risk_score', nbins=20,
                                      title='Rozkład Ryzyka Churnu',
                                      labels={'churn_risk_score': 'Ryzyko Churnu (0-1)'})
        st.plotly_chart(fig_churn_dist, use_container_width=True)


# --- Sekcja: Klienci & Scoring ---
elif selected_option == "Klienci & Scoring":
    st.header("Lista Klientów i Ich Scoring")
    st.write("Wyszukaj klientów i zobacz szczegółowe informacje o ich potencjale sprzedażowym i ryzyku odejścia.")

    search_query = st.text_input("Wyszukaj klienta po ID lub segmencie:", "")
    filtered_df = df[df['client_id'].str.contains(search_query, case=False) | 
                     df['segment'].str.contains(search_query, case=False)]

    st.dataframe(filtered_df[['client_id', 'segment', 'country', 'clinic_size', 'devices_owned', 
                               'upsell_propensity_score', 'churn_risk_score']].style.format({
                                   'upsell_propensity_score': "{:.2f}", 
                                   'churn_risk_score': "{:.2f}"
                               }).background_gradient(cmap='Greens', subset=['upsell_propensity_score']) # Zielony dla wysokiej sprzedazy
                               .background_gradient(cmap='Reds', subset=['churn_risk_score']), # Czerwony dla wysokiego churnu
                               hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Szczegóły Klienta i Rekomendacje")
    
    selected_client_id = st.selectbox(
        "Wybierz ID klienta, aby zobaczyć szczegóły:",
        options=['Wybierz klienta'] + filtered_df['client_id'].tolist()
    )

    if selected_client_id != 'Wybierz klienta':
        client_data = df[df['client_id'] == selected_client_id].iloc[0]
        
        st.write(f"### Dane Klienta: {client_data['client_id']} ({client_data['segment']} - {client_data['country']})")
        st.write(f"**Wielkość kliniki:** {int(client_data['clinic_size'])} | **Urządzeń:** {int(client_data['devices_owned'])}")
        st.write(f"**Ostatni zakup:** {int(client_data['last_purchase_days_ago'])} dni temu | **Liczba zakupów:** {int(client_data['purchase_count'])} | **Średnia wartość zakupu:** {client_data['avg_purchase_value']:.2f} USD")
        st.write(f"**TU2 Aktywne:** {'Tak' if client_data['tu2_active'] else 'Nie'} | **Sesje TU2 (ostatnie 30 dni):** {int(client_data['tu2_sessions_last_30d'])} | **Wykorzystanie AI:** {client_data['ai_usage_ratio']:.0%}")
        st.write(f"**Ostatni kontakt:** {int(client_data['last_contact_days_ago'])} dni temu | **Open Rate:** {client_data['open_rate']:.0%} | **Click Rate:** {client_data['click_rate']:.0%}")
        st.write(f"**Zgłoszenia Supportowe (ostatnie 6m):** {int(client_data['support_tickets_last_6m'])}")

        st.markdown("---")

        col_scores, col_recommendations = st.columns(2)
        with col_scores:
            st.metric("Potencjał Sprzedażowy (Upsell Propensity)", f"{client_data['upsell_propensity_score']:.2f}", 
                      delta_color="off", help="Wysoka wartość (bliżej 1.0) oznacza dużą szansę na dosprzedaż.")
            st.metric("Ryzyko Churnu (Churn Risk)", f"{client_data['churn_risk_score']:.2f}", 
                      delta_color="off", help="Wysoka wartość (bliżej 1.0) oznacza duże ryzyko odejścia klienta.")
        
        with col_recommendations:
            recommendation = get_product_recommendation(client_data)
            contact_strategy = get_contact_strategy(client_data)
            conversation_script = get_conversation_script(client_data, recommendation)

            st.markdown(f"**Sugerowana forma kontaktu:** {contact_strategy}")
            st.markdown(f"**Rekomendowany produkt/działanie:** {recommendation}")
            st.markdown(f"**Sugerowany skrypt rozmowy:**")
            st.markdown(conversation_script)

        st.markdown("---")
        st.subheader("Wpływ Cech na Wynik Modelu (SHAP - Symulacja)")
        st.write("Wykres poniżej pokazuje, które cechy klienta miały największy wpływ na jego scoring.")

        # Wykres SHAP dla Potencjału Sprzedażowego
        plot_simulated_shap(features_for_models, client_data, shap_map_buy, 
                            title=f"Wpływ cech na Potencjał Sprzedażowy ({client_data['client_id']})")
        
        # Wykres SHAP dla Ryzyka Churnu
        plot_simulated_shap(features_for_models, client_data, shap_map_churn, 
                            title=f"Wpływ cech na Ryzyko Churnu ({client_data['client_id']})")

# --- Sekcja: Analiza Modeli AI ---
elif selected_option == "Analiza Modeli AI":
    st.header("Dashboard Analizy Modeli AI")
    st.write("Przegląd ogólnej skuteczności i działania modeli predykcyjnych.")

    st.subheader("Rozkład Potencjału Sprzedażowego wg Segmentu")
    fig_upsell_segment = px.box(df, x='segment', y='upsell_propensity_score', 
                                title='Potencjał Sprzedażowy wg Segmentu Klienta',
                                color='segment',
                                labels={'upsell_propensity_score': 'Potencjał Sprzedażowy', 'segment': 'Segment'})
    st.plotly_chart(fig_upsell_segment, use_container_width=True)

    st.subheader("Rozkład Ryzyka Churnu wg Segmentu")
    fig_churn_segment = px.box(df, x='segment', y='churn_risk_score', 
                               title='Ryzyko Churnu wg Segmentu Klienta',
                               color='segment',
                               labels={'churn_risk_score': 'Ryzyko Churnu', 'segment': 'Segment'})
    st.plotly_chart(fig_churn_segment, use_container_width=True)

    st.markdown("---")
    st.subheader("Korelacja między aktywnością TU2 a Churnem")
    fig_scatter = px.scatter(df, x='tu2_sessions_last_30d', y='churn_risk_score', 
                             color='segment', hover_name='client_id',
                             title='Sesje TU2 vs Ryzyko Churnu',
                             labels={'tu2_sessions_last_30d': 'Sesje TU2 (ostatnie 30 dni)', 'churn_risk_score': 'Ryzyko Churnu'})
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("Ważność Cech dla Modeli (Globalna)")
    st.write("Oto symulowany globalny ranking ważności cech, które najbardziej wpływają na decyzje modeli (XGBoost Feature Importance).")
    
    # Symulacja Feature Importance (w prawdziwym modelu byłoby model.feature_importances_)
    st.write("### Model Potencjału Sprzedażowego")
    buy_importance_df = pd.DataFrame({
        'Cecha': ['tu2_sessions_last_30d', 'purchase_count', 'devices_owned', 'avg_purchase_value', 'click_rate', 'segment_encoded', 'clinic_size'],
        'Ważność': [0.25, 0.20, 0.15, 0.12, 0.10, 0.09, 0.08]
    }).sort_values(by='Ważność', ascending=False)
    fig_buy_imp = px.bar(buy_importance_df, x='Ważność', y='Cecha', orientation='h',
                         title='Ważność Cech - Potencjał Sprzedażowy')
    st.plotly_chart(fig_buy_imp, use_container_width=True)

    st.write("### Model Ryzyka Churnu")
    churn_importance_df = pd.DataFrame({
        'Cecha': ['tu2_sessions_last_30d', 'last_contact_days_ago', 'support_tickets_last_6m', 'tu2_active', 'ai_usage_ratio', 'last_purchase_days_ago'],
        'Ważność': [0.30, 0.20, 0.15, 0.12, 0.10, 0.08]
    }).sort_values(by='Ważność', ascending=False)
    fig_churn_imp = px.bar(churn_importance_df, x='Ważność', y='Cecha', orientation='h',
                           title='Ważność Cech - Ryzyko Churnu')
    st.plotly_chart(fig_churn_imp, use_container_width=True)

# --- Sekcja: Opis Projektu & POC ---
elif selected_option == "Opis Projektu & POC":
    st.header("Opis Projektu Dyplomowego Vet-Eye S.A. & Proof of Concept (POC)")

    st.subheader("Podsumowanie Projektu")
    st.markdown("""
    Projekt **Inteligentny CRM dla Vet-Eye S.A.** to wdrożenie rozwiązań AI w celu poprawy efektywności sprzedaży, zwiększenia retencji klientów (churn prediction) oraz optymalizacji działań handlowców (lead scoring). Jest to praktyczne wdrożenie wiedzy zdobytej na studiach podyplomowych "Biznes.AI" w Akademii Leona Koźmińskiego.
    """)

    st.subheader("Cel POC i Demonstratora")
    st.markdown("""
    Celem **Proof of Concept (POC)** jest weryfikacja kluczowych założeń technicznych i biznesowych systemu AI. **Demonstrator**, który Państwo właśnie oglądają, to działająca aplikacja Streamlit, która wizualizuje funkcjonalności predykcyjne (scoring sprzedażowy i antychurnowy) oraz rekomendacje dla handlowców, w oparciu o syntetyczne dane CRM.
    """)

    st.subheader("1. Proces pozyskania danych niezbędnych do stworzenia systemu AI")
    st.markdown("""
    * **Specyfikacja potrzebnych danych:** System wykorzystuje dane z istniejącego CRM Vet-Eye S.A., takie jak: historia zakupów, aktywność w platformie TU2 (sesje, użycie AI), dane kontaktowe, informacje o zgłoszeniach serwisowych oraz podstawowe dane demograficzne klientów (segment, wielkość kliniki, posiadane urządzenia).
    * **Opis danych dostępnych:** Wszystkie wykorzystywane dane są symulowane na potrzeby demonstratora i pochodzą z wewnętrznego systemu CRM firmy. Nie korzystano z zewnętrznych zbiorów danych open-source.
    * **Metodologia zbierania danych:** Dane są zbierane przez system CRM w sposób ciągły. Na potrzeby modelu są agregowane, czyszczone i przygotowywane, aby służyć jako wejście do algorytmów uczenia maszynowego.
    """)
    with st.expander("Dlaczego zrezygnowaliśmy z cechy 'country' w modelu?"):
        st.markdown("""
        W początkowej fazie projektu, zmienna `country` (kraj) była brana pod uwagę jako potencjalny predyktor. Jednakże, z uwagi na **dominującą większość danych pochodzących z rynku polskiego** w dostępnym zbiorze danych (co jest zgodne z założeniem pilotażu na rynku polskim), oraz **potencjalne niespójności w jakości danych dla pozostałych, słabo reprezentowanych rynków**, zdecydowaliśmy się wykluczyć ją z bieżącej wersji modeli XGBoost.
        
        Takie podejście pozwala nam skupić się na najsilniejszych predyktorach i zapewnia większą stabilność modelu dla głównego rynku, jednocześnie podkreślając potrzebę **rozbudowy i standaryzacji danych międzynarodowych w przyszłych fazach projektu**, jeśli Vet-Eye S.A. będzie rozszerzać swoje działania AI globalnie. Jest to przykład **pragmatycznego zarządzania zakresem i jakością danych** w projekcie Data Science.
        """)


    st.subheader("2. Wybór i opis odpowiedniej technologii")
    st.markdown("""
    * **Hardware (in-house vs cloud):**
        * **Wybór Vet-Eye S.A.:** Firma Vet-Eye S.A. korzysta z **dzierżawionych centrów danych (in-house)**, co zapewnia wysoką dostępność (99,997%) i pełną kontrolę nad danymi. Instalacja AI będzie realizowana przez firmę zewnętrzną.
        * **Uzasadnienie:** In-house (lub dzierżawa) jest preferowane ze względu na **bezpieczeństwo i zgodność z RODO/AI Act**, kluczową w sektorze medycznym. Zapewnia to również niższą latencję i potencjalnie niższe koszty w długoterminowej perspektywie przy dużej skali danych i ciągłym użyciu.
        * **Dla POC:** Demonstrator działa na platformie Streamlit Community Cloud, symulując dostęp do danych.
    * **Metody (frameworki, biblioteki):**
        * **Język:** Python (standard w AI/ML).
        * **Aplikacja webowa/Demonstrator:** **Streamlit** (szybkie prototypowanie interaktywnych aplikacji).
        * **Biblioteki AI/ML:** **XGBoost** (algorytm predykcyjny), **Pandas** (przetwarzanie danych), **NumPy** (operacje numeryczne), **Scikit-learn** (narzędzia ML).
        * **Wizualizacja:** **Plotly** (interaktywne wykresy).
    * **Architektura systemu (blokowy schemat):**
        1.  **Moduł Pozyskiwania Danych (CRM Data Extraction):** Symulacja pobierania danych z CRM.
        2.  **Moduł Przetwarzania Danych (Data Preprocessing):** Czyszczenie, kodowanie, inżynieria cech.
            * *Biblioteki:* Pandas, Scikit-learn (LabelEncoder).
        3.  **Model AI - Scoring Sprzedażowy:** Wytrenowany model klasyfikujący.
            * *Biblioteki/Metody:* **XGBoost (XGBClassifier)**.
        4.  **Model AI - Scoring Antychurnowy:** Wytrenowany model klasyfikujący.
            * *Biblioteki/Metody:* **XGBoost (XGBClassifier)**.
        5.  **Moduł Logiki Eksperckiej (Rule-Based Recommendations):** Generowanie rekomendacji produktów i skryptów rozmów.
            * *Technologia:* Python (warunki logiczne).
        6.  **Interfejs Użytkownika (Handlowca):** Wizualizacja scoringów, rekomendacji, dashboardów.
            * *Technologia:* **Streamlit**, Plotly.
        7.  **Baza Danych (POC):** Plik `data.csv` (symulacja danych CRM).
    """)
    with st.expander("Dlaczego dwa niezależne modele AI, a nie jeden?"):
        st.markdown("""
        Decyzja o zastosowaniu **dwóch niezależnych modeli XGBoost** (dla sprzedaży i antychurnu) zamiast jednego, bardziej złożonego modelu, jest **całkowicie zasadna** dla Vet-Eye S.A., zwłaszcza na etapie POC i wczesnego wdrożenia, ponieważ:
        * **Różne Cele Biznesowe:** Każdy model adresuje odrębny cel – **zwiększenie przychodów** (sprzedaż) vs. **zmniejszenie strat** (retencja). Mieszanie tych celów w jednym modelu mogłoby obniżyć precyzję.
        * **Różne Zmienne Celu:** Modele przewidują dwie odrębne zmienne (`buy_label` i `churn_label`). Standardowe algorytmy klasyfikacji nie obsługują tego efektywnie w ramach jednego modelu.
        * **Jasna Interpretacja dla Handlowców:** Handlowcy otrzymują **dwie jasne, niezależne informacje**: "Ten klient ma potencjał do zakupu" i "Ten klient jest zagrożony odejściem". To daje im pełniejszy i mniej dwuznaczny obraz sytuacji.
        * **Prostsza Architektura i Utrzymanie:** Dwa mniejsze modele są łatwiejsze do zaimplementowania i niezależnego rozwoju. Pozwala to na szybsze wprowadzanie zmian i rozwiązywanie problemów w przyszłości, bez wpływu na drugi model.
        * **Różne Zestawy Cech Kluczowych:** Mimo że dane wejściowe są wspólne, waga poszczególnych cech dla predykcji sprzedaży i churnu może się różnić. Dwa modele pozwalają na optymalizację pod kątem tych różnic.
        """)

    st.subheader("3. Definicja i opis fazy dostosowania (trenowania) i wykorzystania systemu AI")
    st.markdown("""
    * **Proces trenowania:**
        * **Zbieranie danych treningowych:** Agregacja historycznych danych CRM (na potrzeby POC - symulacja z `data.csv`).
        * **Przygotowanie danych:** Czyszczenie, kodowanie zmiennych kategorialnych (np. `segment`).
        * **Podział danych:** Na zbiory treningowe i testowe w celu walidacji.
        * **Trening modeli XGBoost:** Dwa niezależne klasyfikatory XGBoost są trenowane. Algorytm wybrany jest ze względu na wysoką skuteczność i odporność na przeuczenie.
        * **Walidacja:** Ocena modeli na zbiorze testowym.
    * **Szacunkowy czas treningu i koszt:**
        * **Dla POC:** Czas treningu na 1000 rekordów jest **pomijalny** (sekundy). Brak kosztów chmurowych, aplikacja wykorzystuje minimalne zasoby Streamlit Cloud.
        * **Dla realnego wdrożenia:** Czas treningu może wynosić od **kilku minut do kilku godzin** na dedykowanych serwerach Vet-Eye S.A., w zależności od wolumenu danych (miliony rekordów) i złożoności cech. Koszty związane z amortyzacją i utrzymaniem infrastruktury in-house, a nie z opłatami za publiczną chmurę.
    """)

    st.subheader("4. Ocena efektywności systemu AI")
    st.markdown("""
    * **Techniczna:**
        * **Metryki:** Dokładność (Accuracy), Precyzja (Precision), Czułość (Recall), F1-score, AUC ROC – stosowane do oceny poprawności predykcji.
        * **Szybkość inferencji:** Na potrzeby POC inferencja dla pojedynczego klienta jest **natychmiastowa** (milisekundy). W realnym systemie, przetwarzanie całej bazy jest regularne i szybkie.
        * **Funkcjonalna:** Intuicyjność interfejsu Streamlit, łatwość dostępu do kluczowych informacji i rekomendacji dla handlowców.
        * **Niefunkcjonalna:** Skalowalność (zdolność do obsługi rosnącej liczby klientów), niezawodność (wysoka dostępność dzięki dwóm centrom danych Vet-Eye S.A.), bezpieczeństwo (zgodność z RODO i AI Act).
    * **Biznesowa (ROI/Break-even):**
        * **Wzrost przychodów:** Poprawa współczynnika konwersji leadów (dzięki lepszemu lead scoringowi) i zwiększenie średniej wartości transakcji (dzięki rekomendacjom upsellowym).
        * **Zmniejszenie kosztów:** Redukcja odpływu klientów (dzięki predykcji churnu i proaktywnym działaniom retencyjnym), optymalizacja czasu pracy handlowców.
        * **Break-even Point:** Punkt, w którym skumulowane korzyści (zwiększone przychody + zmniejszone koszty) zrównają się z kosztami wdrożenia i utrzymania systemu AI. **Na etapie POC są to szacunki i prognozy**, bazujące na hipotetycznych wzrostach wskaźników biznesowych.
    """)