#!/usr/bin/env python3
"""
Enhanced Domain Classification using Multiple SOTA Models
========================================================

This program provides maximum accuracy for domain classification by:
1. Using ensemble of 6 top embedding models (E5, MPNet, MiniLM, LaBSE, Jina, mDeBERTa)
2. Advanced preprocessing and feature engineering 
3. Domain-specific pattern recognition
4. Multi-level confidence scoring
5. Conflict resolution with retry logic (up to 3 attempts)
6. Vague classification detection (only assigns "Other" when truly ambiguous)

Features:
- Excludes "Other" category from initial classification
- Retries domains with tied predictions up to 3 times
- Marks persistent conflicts with asterisk (*)
- Only assigns "Other" for genuinely vague/ambiguous content
- Comprehensive statistics on conflicts and classification quality

Expected Input Structure:
- domain: Clean domain name (e.g., "wikipedia.org", not URLs)
- title: Page title
- meta_description: Meta description
- keywords: Keywords associated with the domain

Usage: python enhanced_domain_classifier.py <input_parquet_file>
"""

import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
import gc
import time
import warnings
import re
from urllib.parse import urlparse
from collections import Counter, defaultdict
import logging
import random

# ML Libraries
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Clustering (if needed in future)
# from bertopic import BERTopic

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDomainClassifier:
    """Enhanced multi-model domain classifier with BERTopic"""
    
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.models = {}
        self.category_embeddings = {}
        self.tld_patterns = {}
        self.domain_features = {}
        
        # Enhanced category definitions with Greek internal names and English output mapping
        self.categories = {
            "Ηλεκτρονικό Εμπόριο & Αγορές": {
                "english_name": "E-Commerce & Shopping",
                "description": "ηλεκτρονικές αγορές, ηλεκτρονικό εμπόριο, λιανικό εμπόριο, αγορά, αγοραπωλησία, πώληση, προϊόντα, καταστήματα, μαγαζιά, καλάθι αγορών, ολοκλήρωση παραγγελίας, πληρωμή, παραγγελίες, ηλεκτρονικό κατάστημα, eshop, αγορές, κατάστημα, πωλήσεις, προϊόντα, εμπόριο, αγορά, πληρωμή, παραγγελία, καλάθι, έμπορος, προμηθευτής, εμπόριο, συναλλαγές, επιχειρήσεις, πωλήσεις, πελάτης, αγορά, συναλλαγή, online shopping, retail, marketplace, merchant, vendor, checkout, payment, cart, order, purchase, sale, discount, deal, offer, coupon, price, product, item, inventory, catalog, store, shop, business, commercial, trade, selling, buying",
                "keywords": ["αγορές", "κατάστημα", "πωλήσεις", "προϊόντα", "εμπόριο", "αγορά", "πληρωμή", "παραγγελία", "καλάθι", "eshop", "ηλεκτρονικό", "μαγαζί", "αγοραπωλησία", "συναλλαγή", "πελάτης", "έμπορος", "προμηθευτής", "τιμή", "προσφορά", "εκπτώσεις", "λιανικό", "χονδρικό", "marketplace", "πλατφόρμα", "shop", "store", "buy", "sell", "ecommerce", "shopping", "retail", "marketplace", "merchant", "vendor", "checkout", "payment", "cart", "order", "purchase", "sale", "discount", "deal", "offer", "coupon", "price", "product", "item", "inventory", "catalog", "business", "commercial", "trade", "selling", "buying", "online", "market", "bazaar", "outlet", "boutique", "supermarket", "mall", "department", "wholesale", "dropship", "fulfillment", "logistics", "shipping", "delivery"],
                "tlds": [".shop", ".store", ".buy", ".sale", ".market", ".deals"],
                "domains": ["amazon", "ebay", "alibaba", "shopify", "etsy", "walmart", "target", "skroutz", "bestprice", "plaisio", "aliexpress", "wish", "overstock", "newegg", "bestbuy", "costco", "homedepot", "lowes", "macys", "nordstrom", "zappos", "wayfair", "ikea", "zara", "hm", "uniqlo", "nike", "adidas", "sephora", "ulta", "cvs", "walgreens", "pharmacy", "grocery", "food", "restaurant", "delivery", "uber", "lyft", "doordash", "grubhub", "instacart", "postmates"]
            },
            "Ειδήσεις & Μέσα Ενημέρωσης": {
                "english_name": "News & Media",
                "description": "ειδήσεις, δημοσιογραφία, εφημερίδες, περιοδικά, μέσα ενημέρωσης, τύπος, ραδιοτηλεόραση, τηλεόραση, ραδιόφωνο, άρθρα, δημοσιογράφοι, έκτακτα νέα, τίτλοι, τρέχοντα γεγονότα, πολιτική, παγκόσμια νέα, τοπικά νέα, ενημέρωση, ρεπορτάζ, συνέντευξη, αναλύσεις, σχολιασμός, μετάδοση, εκπομπές, δελτία ειδήσεων, κανάλια, δημοσιογραφικό έργο, ρεπόρτερ, breaking news, current events, headlines, journalism, reporter, correspondent, editorial, opinion, commentary, broadcast, television, radio, magazine, newspaper, press release, media outlet, news agency, newsroom, anchor, journalist, editor, publisher",
                "keywords": ["ειδήσεις", "εφημερίδα", "νέα", "δημοσιογραφία", "ενημέρωση", "τηλεόραση", "ραδιόφωνο", "άρθρα", "δημοσιογράφος", "πολιτική", "τίτλοι", "ρεπορτάζ", "έκτακτα", "κανάλι", "εκπομπή", "δελτίο", "σχολιασμός", "ανάλυση", "συνέντευξη", "μετάδοση", "περιοδικό", "τύπος", "ρεπόρτερ", "αρθρογράφος", "συντάκτης", "εκδότης", "news", "media", "press", "journalism", "newspaper", "magazine", "reporter", "journalist", "breaking", "headline", "story", "article", "editorial", "opinion", "commentary", "broadcast", "television", "radio", "correspondent", "anchor", "editor", "publisher", "newsroom", "agency", "outlet", "current", "events", "politics", "world", "local", "national", "international", "daily", "weekly", "monthly"],
                "tlds": [".news", ".press", ".media"],
                "domains": ["cnn", "bbc", "reuters", "nytimes", "guardian", "kathimerini", "iefimerida", "newsbeast", "news247", "cnn.gr", "protothema", "tovima", "naftemporiki", "skai", "ap", "afp", "bloomberg", "wsj", "ft", "economist", "time", "newsweek", "usatoday", "washingtonpost", "latimes", "telegraph", "independent", "mirror", "sun", "dailymail", "foxnews", "msnbc", "abc", "cbs", "nbc", "espn", "sky", "itv", "channel4"]
            },
            "Κοινωνικά Δίκτυα & Κοινότητα": {
                "english_name": "Social Media & Community",
                "description": "κοινωνικά δίκτυα, κοινωνικά μέσα, κοινότητα, φόρουμ, συζήτηση, συνομιλία, μηνύματα, δικτύωση, κοινωνική αλληλεπίδραση, περιεχόμενο χρηστών, κοινοποίηση, αναρτήσεις, σχόλια, likes, ακόλουθοι, φίλοι, συνδέσεις, ομάδες, κοινότητες, επικοινωνία, διαμοιρασμός, συζητήσεις, διαδραστικότητα, χρήστες, προφίλ, δημοσιεύσεις, αλληλεπίδραση, δίκτυο επαφών",
                "keywords": ["κοινωνικά", "κοινότητα", "φόρουμ", "συζήτηση", "μηνύματα", "δικτύωση", "κοινοποίηση", "αναρτήσεις", "σχόλια", "φίλοι", "ακόλουθοι", "ομάδες", "επικοινωνία", "διαμοιρασμός", "συνομιλία", "προφίλ", "δημοσιεύσεις", "αλληλεπίδραση", "χρήστες", "δίκτυο", "social", "community", "forum", "chat"],
                "tlds": [".social", ".community", ".forum"],
                "domains": ["facebook", "twitter", "instagram", "linkedin", "reddit", "discord", "telegram", "whatsapp", "tiktok", "snapchat", "youtube"]
            },
            "Τεχνολογία & Λογισμικό": {
                "english_name": "Technology & Software",
                "description": "τεχνολογία, λογισμικό, προγραμματισμός, ανάπτυξη, πληροφορική, υπολογιστές, εφαρμογές, ψηφιακά εργαλεία, τεχνολογικές εταιρείες, κώδικας, ανάπτυξη λογισμικού, υλικό, ηλεκτρονικά, καινοτομία, νεοφυής επιχείρηση, τεχνολογικά νέα, τεχνητή νοημοσύνη, μηχανική μάθηση, δεδομένα, cloud, πλατφόρμα, διαδίκτυο, ιστοσελίδες, συστήματα, βάσεις δεδομένων, δίκτυα, ασφάλεια, κυβερνοασφάλεια, mobile apps, web development, APIs, frameworks, libraries, databases, servers, hosting, SaaS, DevOps, automation, artificial intelligence, machine learning, data science, cybersecurity, blockchain, cryptocurrency, fintech, startup, innovation, digital transformation",
                "keywords": ["τεχνολογία", "λογισμικό", "προγραμματισμός", "ανάπτυξη", "υπολογιστές", "εφαρμογές", "κώδικας", "ηλεκτρονικά", "καινοτομία", "πληροφορική", "ψηφιακά", "τεχνητή", "νοημοσύνη", "δεδομένα", "cloud", "πλατφόρμα", "διαδίκτυο", "ιστοσελίδες", "συστήματα", "βάσεις", "δίκτυα", "ασφάλεια", "κυβερνοασφάλεια", "εφαρμογή", "πλατφόρμες", "servers", "hosting", "tech", "software", "programming", "development", "coding", "developer", "engineer", "computer", "digital", "technology", "innovation", "startup", "app", "application", "platform", "framework", "library", "database", "server", "cloud", "api", "saas", "devops", "automation", "artificial", "intelligence", "machine", "learning", "data", "science", "cybersecurity", "security", "blockchain", "cryptocurrency", "fintech", "mobile", "web", "internet", "online", "digital", "electronic", "hardware", "software", "system", "network", "infrastructure", "solution", "service", "tool", "product"],
                "tlds": [".tech", ".app", ".dev", ".ai", ".io", ".software", ".cloud", ".digital"],
                "domains": ["github", "stackoverflow", "microsoft", "google", "apple", "amazon", "facebook", "netflix", "adobe", "oracle", "ibm", "intel", "cisco", "salesforce", "slack", "zoom", "dropbox", "atlassian", "jetbrains", "gitlab", "bitbucket", "heroku", "aws", "azure", "gcp", "digitalocean", "linode", "vultr", "cloudflare", "fastly", "stripe", "paypal", "twilio", "sendgrid", "mailgun", "firebase", "mongodb", "redis", "postgresql", "mysql", "docker", "kubernetes", "jenkins", "travis", "circleci", "npm", "yarn", "composer", "pip", "maven", "gradle"]
            },
            "Ψυχαγωγία & Μέσα": {
                "english_name": "Entertainment & Media",
                "description": "ψυχαγωγία, ταινίες, μουσική, παιχνίδια, gaming, διασημότητες, τέχνες, πολιτισμός, διασκέδαση, ελεύθερος χρόνος, εκπομπές, συναυλίες, φεστιβάλ, streaming, βίντεο, ήχος, πολυμέσα, δημιουργικό περιεχόμενο, κινηματογράφος, θέατρο, τηλεοπτικές σειρές, ντοκιμαντέρ, podcast, ραδιοφωνικές εκπομπές, καλλιτέχνες, μουσικοί, ηθοποιοί, σκηνοθέτες, παραγωγοί, διασκέδαση, hobby, χόμπι, video games, movies, films, cinema, television, TV shows, series, music, songs, albums, artists, bands, concerts, festivals, comedy, humor, celebrities, gossip, entertainment news, pop culture, lifestyle, fashion, beauty, art, culture, creative, content, media, streaming, video, audio, multimedia, digital content, online entertainment",
                "keywords": ["ψυχαγωγία", "ταινίες", "μουσική", "παιχνίδια", "διασημότητες", "τέχνες", "πολιτισμός", "διασκέδαση", "εκπομπές", "συναυλίες", "φεστιβάλ", "streaming", "βίντεο", "ήχος", "κινηματογράφος", "θέατρο", "σειρές", "ντοκιμαντέρ", "podcast", "καλλιτέχνες", "μουσικοί", "ηθοποιοί", "χόμπι", "gaming", "παιχνίδια", "κωμωδία", "χιούμορ", "lifestyle", "μόδα", "ομορφιά", "entertainment", "music", "movies", "games", "gaming", "films", "cinema", "television", "tv", "shows", "series", "songs", "albums", "artists", "bands", "concerts", "festivals", "comedy", "humor", "celebrities", "gossip", "pop", "culture", "lifestyle", "fashion", "beauty", "art", "creative", "content", "media", "streaming", "video", "audio", "multimedia", "digital", "online", "fun", "leisure", "hobby", "recreation", "amusement", "enjoyment", "pleasure", "relaxation"],
                "tlds": [".entertainment", ".music", ".game", ".tv", ".video", ".media", ".fun"],
                "domains": ["netflix", "youtube", "spotify", "twitch", "steam", "imdb", "hulu", "disney", "warner", "sony", "universal", "paramount", "hbo", "showtime", "starz", "amazon", "prime", "apple", "music", "pandora", "soundcloud", "vimeo", "dailymotion", "metacritic", "rottentomatoes", "fandango", "ticketmaster", "stubhub", "eventbrite", "buzzfeed", "tmz", "eonline", "people", "usmagazine", "rollingstone", "billboard", "variety", "hollywoodreporter", "deadline", "ign", "gamespot", "polygon", "kotaku", "destructoid", "pcgamer", "gameinformer"]
            },
            "Εκπαίδευση & Έρευνα": {
                "english_name": "Education & Research",
                "description": "εκπαίδευση, σχολεία, πανεπιστήμια, μάθηση, ακαδημαϊκά, έρευνα, επιστήμη, σπουδές, μαθήματα, κατάρτιση, γνώση, εκπαιδευτικοί πόροι, βιβλιοθήκες, ακαδημαϊκά ιδρύματα, επιστημονικό περιεχόμενο, διδασκαλία, μάθηση, φοιτητές, καθηγητές, ερευνητές, σπουδαστές, εκπαιδευτικά προγράμματα, σεμινάρια, διαλέξεις, εργαστήρια, πτυχία, μεταπτυχιακά, διδακτορικά, επιστημονικές δημοσιεύσεις, εγκυκλοπαίδεια, λεξικό, αναφορά, πληροφορίες, γνώση, μάθηση, εκπαιδευτικό υλικό, ακαδημαϊκές πηγές, ερευνητικά κέντρα, επιστημονικά περιοδικά, διδακτικό υλικό, online courses, tutorials, reference, documentation",
                "keywords": ["εκπαίδευση", "σχολείο", "πανεπιστήμιο", "μάθηση", "ακαδημαϊκά", "έρευνα", "επιστήμη", "σπουδές", "μαθήματα", "γνώση", "βιβλιοθήκη", "καθηγητής", "φοιτητής", "διδασκαλία", "κατάρτιση", "σεμινάρια", "διαλέξεις", "εργαστήρια", "πτυχία", "μεταπτυχιακά", "διδακτορικά", "ερευνητές", "εγκυκλοπαίδεια", "λεξικό", "αναφορά", "πληροφορίες", "ακαδημαϊκές", "ερευνητικά", "επιστημονικά", "διδακτικό", "εκπαιδευτικό", "education", "university", "research", "science", "wikipedia", "wiki", "encyclopedia", "dictionary", "reference", "academic", "scholar", "learning", "knowledge", "tutorial", "course", "study", "library", "documentation", "manual", "guide", "textbook", "journal", "publication", "thesis", "dissertation", "paper", "article", "scholarly"],
                "tlds": [".edu", ".ac", ".university", ".school", ".academy"],
                "domains": ["mit", "harvard", "stanford", "cambridge", "oxford", "coursera", "edx", "khan", "udemy", "skillshare", "uoa", "ntua", "auth", "upatras", "wikipedia", "wiki", "wikimedia", "wiktionary", "britannica", "encyclopedia", "dictionary", "reference", "scholar", "researchgate", "academia", "arxiv", "pubmed", "jstor", "springer", "elsevier", "nature", "science", "ieee", "acm"]
            },
            "Υγεία & Ιατρική": {
                "english_name": "Health & Medical",
                "description": "υγεία, ιατρική, υγειονομική περίθαλψη, νοσοκομεία, γιατροί, φάρμακα, ευεξία, φυσική κατάσταση, φαρμακείο, ιατρικές υπηρεσίες, πληροφορίες υγείας, ασθένεια, θεραπεία, διάγνωση, ιατρική έρευνα, κλινικές, ιατροί, νοσηλευτές, φαρμακευτικά, ιατρικά εργαλεία, χειρουργική, πρόληψη, ιατρική φροντίδα, ψυχική υγεία, οδοντιατρική, παιδιατρική, γυναικολογία, καρδιολογία, ορθοπεδική, δερματολογία",
                "keywords": ["υγεία", "ιατρική", "νοσοκομείο", "γιατρός", "φάρμακα", "ευεξία", "φαρμακείο", "κλινική", "θεραπεία", "διάγνωση", "ασθένεια", "ασθενής", "φροντίδα", "ιατροί", "νοσηλευτές", "φαρμακευτικά", "χειρουργική", "πρόληψη", "ψυχική", "οδοντιατρική", "παιδιατρική", "health", "medical", "doctor", "medicine"],
                "tlds": [".health", ".medical", ".care", ".clinic"],
                "domains": ["webmd", "mayoclinic", "healthline", "medscape", "nih", "who", "cdc", "medline", "pubmed"]
            },
            "Κυβέρνηση & Δημόσιες Υπηρεσίες": {
                "english_name": "Government & Public Services",
                "description": "κυβέρνηση, δημόσιες υπηρεσίες, διοίκηση, πολιτική, επίσημα, κράτος, ομοσπονδιακά, τοπική κυβέρνηση, δημόσιος τομέας, πολιτικά, κυβερνητικά, επίσημα ιδρύματα, υπουργεία, δήμοι, περιφέρειες, κυβερνητικοί οργανισμοί, δημόσια διοίκηση, νομοθεσία, κανονισμοί, δημόσιες πολιτικές, εκλογές, πολιτικά κόμματα, βουλή, κοινοβούλιο, δικαστήρια, αστυνομία, στρατός, δημόσια ασφάλεια",
                "keywords": ["κυβέρνηση", "δημόσιες", "επίσημα", "κράτος", "διοίκηση", "πολιτική", "πολιτικά", "υπουργείο", "τμήμα", "οργανισμός", "γραφείο", "δήμος", "περιφέρεια", "νομοθεσία", "κανονισμοί", "εκλογές", "κόμματα", "βουλή", "κοινοβούλιο", "δικαστήρια", "αστυνομία", "στρατός", "government", "public", "official"],
                "tlds": [".gov", ".mil", ".pol", ".gr"],
                "domains": ["usa", "uk", "europa", "un", "who", "nato", "government", "gov", "ypes", "minedu", "minhealth"]
            },
            "Ταξίδια & Τουρισμός": {
                "english_name": "Travel & Tourism",
                "description": "ταξίδια, τουρισμός, ξενοδοχεία, πτήσεις, διακοπές, προορισμοί, κράτηση, φιλοξενία, διαμονή, εστιατόρια, σχεδιασμός ταξιδιού, περιπέτεια, εξερεύνηση, τουριστικά αξιοθέατα, ταξιδιωτικοί οδηγοί, αεροπορικά εισιτήρια, κρουαζιέρες, ταξιδιωτικά πακέτα, τουριστικοί προορισμοί, διακοπές, εκδρομές, ταξιδιωτικές υπηρεσίες, ταξιδιωτικά γραφεία, ξενώνες, resort, camping, backpacking",
                "keywords": ["ταξίδια", "τουρισμός", "ξενοδοχείο", "πτήση", "διακοπές", "προορισμός", "κράτηση", "φιλοξενία", "διαμονή", "εστιατόριο", "ταξίδι", "περιπέτεια", "εξερεύνηση", "οδηγός", "αξιοθέατα", "αεροπορικά", "κρουαζιέρες", "πακέτα", "εκδρομές", "ξενώνες", "resort", "camping", "travel", "tourism", "hotel", "flight"],
                "tlds": [".travel", ".hotel", ".vacation", ".tour"],
                "domains": ["booking", "expedia", "airbnb", "tripadvisor", "hotels", "kayak", "priceline", "agoda", "trivago"]
            },
            "Χρηματοοικονομικά & Τραπεζικά": {
                "english_name": "Finance & Banking",
                "description": "χρηματοοικονομικά, τραπεζικά, χρηματοοικονομικές υπηρεσίες, επενδύσεις, ασφάλιση, κρυπτονομίσματα, fintech, χρήματα, πληρωμή, πίστωση, δάνεια, οικονομικά, αγορά, συναλλαγές, τράπεζες, χρηματιστήριο, μετοχές, ομόλογα, χρηματοδότηση, οικονομική ανάλυση, λογιστικά, φορολογικά, ασφαλιστικές εταιρείες, χρηματοπιστωτικά ιδρύματα, οικονομικοί σύμβουλοι, investment banking, private banking",
                "keywords": ["χρηματοοικονομικά", "τραπεζικά", "επενδύσεις", "ασφάλιση", "χρήματα", "πληρωμή", "πίστωση", "δάνεια", "οικονομικά", "αγορά", "συναλλαγές", "τράπεζες", "χρηματιστήριο", "μετοχές", "ομόλογα", "χρηματοδότηση", "λογιστικά", "φορολογικά", "ασφαλιστικές", "χρηματοπιστωτικά", "σύμβουλοι", "finance", "banking", "investment", "insurance"],
                "tlds": [".bank", ".finance", ".money", ".insurance"],
                "domains": ["paypal", "visa", "mastercard", "american express", "chase", "wells fargo", "goldman sachs", "morgan stanley", "bloomberg", "reuters", "alphabank", "eurobank", "nbg", "piraeusbank"]
            },
            "Αθλητισμός & Αναψυχή": {
                "english_name": "Sports & Recreation",
                "description": "αθλητισμός, αθλητικά, αναψυχή, φυσική κατάσταση, παιχνίδια, διαγωνισμός, ομάδες, πρωταθλήματα, αθλητές, αθλητικά γεγονότα, υπαίθριες δραστηριότητες, άσκηση, φυσική δραστηριότητα, προπόνηση, γυμναστήρια, αθλητικά κέντρα, αθλητικές εγκαταστάσεις, ποδόσφαιρο, μπάσκετ, τένις, κολύμβηση, τρέξιμο, ποδηλασία, γυμναστική, βόλεϊ, χάντμπολ, αθλητικές ειδήσεις, αθλητικά αποτελέσματα",
                "keywords": ["αθλητισμός", "αθλητικά", "αναψυχή", "φυσική κατάσταση", "παιχνίδια", "διαγωνισμός", "ομάδες", "πρωταθλήματα", "αθλητές", "άσκηση", "φυσική δραστηριότητα", "προπόνηση", "γυμναστήρια", "ποδόσφαιρο", "μπάσκετ", "τένις", "κολύμβηση", "τρέξιμο", "ποδηλασία", "γυμναστική", "βόλεϊ", "χάντμπολ", "sport", "sports", "fitness", "athletic"],
                "tlds": [".sport", ".fitness", ".team"],
                "domains": ["espn", "nba", "nfl", "fifa", "olympics", "sport", "athletic", "fitness", "gazzetta", "sport24", "contra", "novasports"]
            },
            "Άλλο": {
                "english_name": "Other",
                "description": "διάφορα, γενικά, μικτό περιεχόμενο, μη καθορισμένη κατηγορία, διάφορα θέματα, ποικίλο περιεχόμενο, γενικές πληροφορίες, ασαφής σκοπός, ανάμεικτο περιεχόμενο, μη ταξινομημένο, γενικού ενδιαφέροντος, πολλαπλές κατηγορίες, ασαφές περιεχόμενο, μη συγκεκριμένο θέμα, ποικιλία θεμάτων, γενικές υπηρεσίες",
                "keywords": ["γενικά", "διάφορα", "μικτό", "άλλο", "μη καθορισμένο", "ποικίλο", "διαφορετικό", "ανάμεικτο", "ταξινομημένο", "γενικού", "ενδιαφέροντος", "πολλαπλές", "ασαφές", "συγκεκριμένο", "ποικιλία", "γενικές", "υπηρεσίες", "general", "misc", "various", "mixed", "other"],
                "tlds": [],
                "domains": []
            }
        }
        
        # Create Greek to English mapping
        self.greek_to_english = {greek_name: info['english_name'] for greek_name, info in self.categories.items()}
        
        print(f"🚀 Enhanced Domain Classifier initialized on {self.device}")
    
    def _convert_to_english(self, greek_category):
        """Convert Greek category name to English for output"""
        # Handle asterisk prefix for conflicts
        if greek_category.startswith('*'):
            greek_name = greek_category[1:]
            english_name = self.greek_to_english.get(greek_name, greek_name)
            return f"*{english_name}"
        else:
            return self.greek_to_english.get(greek_category, greek_category)
    
    def setup_models(self):
        """Initialize ensemble of embedding models"""
        print("📥 Loading embedding models...")
        
        # Original model configurations - fixed model names
        model_configs = [
            ("e5-large", "intfloat/multilingual-e5-large"),
            ("mpnet", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            ("labse", "sentence-transformers/LaBSE"),
            ("jina", "jinaai/jina-embeddings-v3"),
            ("mdeberta", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
        ]
        
        # Use /mnt/data for cache to avoid disk space issues
        import os
        cache_dir = "/mnt/data/jimmy/huggingface_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variables to force HuggingFace to use our cache directory
        os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        
        successful_models = []
        failed_models = []
        
        for name, model_name in model_configs:
            try:
                print(f"   Loading {name}...")
                
                # Try loading with different strategies
                model = None
                
                if name == "jina":
                    # Try different Jina models
                    jina_models = [
                        "jinaai/jina-embeddings-v3",
                        "jinaai/jina-embeddings-v2-base-en", 
                        "jinaai/jina-embeddings-v2-small-en"
                    ]
                    for jina_model in jina_models:
                        try:
                            print(f"     Trying {jina_model}...")
                            model = SentenceTransformer(jina_model, cache_folder=cache_dir, trust_remote_code=True)
                            break
                        except Exception as e:
                            print(f"     Failed {jina_model}: {str(e)[:100]}...")
                            continue
                            
                elif name == "mdeberta":
                    # Try different mDeBERTa models
                    mdeberta_models = [
                        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                        "microsoft/mdeberta-v3-base",
                        "microsoft/deberta-v3-base"
                    ]
                    for mdeberta_model in mdeberta_models:
                        try:
                            print(f"     Trying {mdeberta_model}...")
                            model = SentenceTransformer(mdeberta_model, cache_folder=cache_dir, trust_remote_code=True)
                            break
                        except Exception as e:
                            print(f"     Failed {mdeberta_model}: {str(e)[:100]}...")
                            continue
                            
                elif name == "labse":
                    # Try different LaBSE models
                    labse_models = [
                        "sentence-transformers/LaBSE",
                        "cointegrated/LaBSE-en-ru"
                    ]
                    for labse_model in labse_models:
                        try:
                            print(f"     Trying {labse_model}...")
                            model = SentenceTransformer(labse_model, cache_folder=cache_dir)
                            break
                        except Exception as e:
                            print(f"     Failed {labse_model}: {str(e)[:100]}...")
                            continue
                else:
                    # Standard loading for other models
                    model = SentenceTransformer(model_name, cache_folder=cache_dir)
                
                if model is not None:
                    model = model.to(self.device)
                    self.models[name] = model
                    successful_models.append(name)
                    print(f"   ✅ {name} loaded successfully")
                else:
                    failed_models.append(name)
                    print(f"   ❌ All attempts failed for {name}")
                    
            except Exception as e:
                failed_models.append(name)
                print(f"   ❌ Failed to load {name}: {str(e)[:100]}...")
                continue
        
        print(f"\n📊 Model Loading Summary:")
        print(f"   ✅ Successfully loaded: {len(successful_models)} models: {', '.join(successful_models)}")
        if failed_models:
            print(f"   ❌ Failed to load: {len(failed_models)} models: {', '.join(failed_models)}")
        
        if not self.models:
            raise RuntimeError("No embedding models could be loaded!")
        
        print(f"✅ Loaded {len(self.models)} embedding models")
        
        # Pre-compute category embeddings for each model
        self._compute_category_embeddings()
    
    def _compute_category_embeddings(self):
        """Pre-compute embeddings for all categories using all models"""
        print("🧮 Computing category embeddings...")
        
        for model_name, model in self.models.items():
            print(f"   Computing embeddings with {model_name}...")
            category_texts = []
            category_names = []
            
            for category, info in self.categories.items():
                # Create rich text representation
                text = f"Category: {category}. Description: {info['description']}. Keywords: {' '.join(info['keywords'][:20])}"
                category_texts.append(text)
                category_names.append(category)
            
            embeddings = model.encode(category_texts, convert_to_tensor=True, device=self.device, show_progress_bar=False)
            self.category_embeddings[model_name] = {
                'embeddings': embeddings,
                'names': category_names
            }
        
        print("✅ Category embeddings computed")
    

    
    def extract_domain_features(self, domain_name):
        """Extract domain-specific features"""
        try:
            domain = domain_name.lower().strip()
            
            # Remove www prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Extract TLD
            tld = domain.split('.')[-1] if '.' in domain else ''
            
            # Extract domain name without TLD and convert dots to spaces
            if '.' in domain:
                domain_parts = domain.split('.')
                base_domain_name = '.'.join(domain_parts[:-1])  # Remove TLD
                processed_domain_name = base_domain_name.replace('.', ' ')  # Replace dots with spaces
            else:
                processed_domain_name = domain
            
            return {
                'domain': domain,
                'domain_name': processed_domain_name,
                'tld': tld
            }
        except:
            return {
                'domain': domain_name,
                'domain_name': domain_name.replace('.', ' '),
                'tld': ''
            }
    
    def preprocess_text(self, row):
        """Enhanced text preprocessing"""
        parts = []
        
        # Extract domain features
        domain_features = self.extract_domain_features(row.get('domain', ''))
        
        # Check if site has any metadata (title, description, keywords)
        has_metadata = False
        metadata_fields = ['title', 'meta_description', 'keywords']
        
        for field in metadata_fields:
            if field in row and pd.notna(row[field]):
                text = str(row[field]).strip()
                if text and text.lower() not in ['none', 'null', 'nan', '']:
                    has_metadata = True
                    parts.append(text)
        
        # Add domain name as separate feature
        if domain_features['domain_name']:
            parts.append(domain_features['domain_name'])
        
        combined_text = ' '.join(parts) if parts else ''
        
        # Store metadata status in domain_features for later use
        domain_features['has_metadata'] = has_metadata
        
        return combined_text, domain_features
    
    def classify_with_ensemble(self, text, domain_features, retry_count=0):
        """Classify using ensemble of models with conflict resolution"""
        if not text:
            return "Άλλο", 0.1
        
        # Automatically assign "Other" if site has no metadata
        if not domain_features.get('has_metadata', True):
            return "Άλλο", 0.1
        
        # Truncate text if too long
        text = text[:3000] if len(text) > 3000 else text
        
        predictions = []
        confidences = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                # Encode text
                text_embedding = model.encode([text], convert_to_tensor=True, device=self.device, show_progress_bar=False)
                
                # Calculate similarities
                category_data = self.category_embeddings[model_name]
                similarities = cosine_similarity(
                    text_embedding.cpu().numpy(),
                    category_data['embeddings'].cpu().numpy()
                )[0]
                
                # Get best prediction (exclude "Other" from initial classification)
                filtered_categories = [cat for cat in category_data['names'] if cat != "Άλλο"]
                filtered_similarities = []
                filtered_names = []
                
                for i, cat_name in enumerate(category_data['names']):
                    if cat_name != "Άλλο":
                        filtered_similarities.append(similarities[i])
                        filtered_names.append(cat_name)
                
                if filtered_similarities:
                    best_idx = np.argmax(filtered_similarities)
                    best_category = filtered_names[best_idx]
                    confidence = float(filtered_similarities[best_idx])
                else:
                    # Fallback if no categories available
                    best_idx = np.argmax(similarities)
                    best_category = category_data['names'][best_idx]
                    confidence = float(similarities[best_idx])
                
                predictions.append(best_category)
                confidences.append(confidence)
                
            except Exception as e:
                logger.warning(f"Error with model {model_name}: {e}")
                continue
        
        if not predictions:
            return "Άλλο", 0.1
        
        # Check for conflicts (ties in predictions)
        prediction_counts = Counter(predictions)
        max_count = max(prediction_counts.values())
        tied_categories = [cat for cat, count in prediction_counts.items() if count == max_count]
        
        # If there's a tie and we haven't exceeded retry limit
        if len(tied_categories) > 1 and retry_count < 3:
            # Add some randomness to break ties by slightly modifying text
            random_words = random.sample(['domain', 'website', 'site', 'content', 'page', 'portal'], 2)
            modified_text = text + f" {' '.join(random_words)}"
            return self.classify_with_ensemble(modified_text, domain_features, retry_count + 1)
        
        # Apply domain-specific rules
        enhanced_prediction = self._apply_domain_rules(predictions, confidences, text, domain_features)
        
        # Check if results are too vague (low confidence across all models)
        avg_confidence = np.mean(confidences)
        if avg_confidence < 0.3 and len(set(predictions)) >= len(predictions) * 0.7:
            # Too many different predictions with low confidence
            return "Άλλο", avg_confidence
        
        # If we had ties and exceeded retry limit, mark with asterisk
        if len(tied_categories) > 1 and retry_count >= 3:
            category, confidence = enhanced_prediction
            return f"*{category}", confidence
        
        return enhanced_prediction
    
    def _apply_domain_rules(self, predictions, confidences, text, domain_features):
        """Apply domain-specific rules to enhance predictions"""
        text_lower = text.lower()
        domain_name = domain_features['domain_name'].lower()
        tld = domain_features['tld'].lower()
        
        # Rule-based enhancements (exclude "Other" from rule matching)
        rule_boosts = defaultdict(float)
        
        # Check for domain name patterns
        for category, info in self.categories.items():
            if category == "Άλλο":
                continue  # Skip "Other" category in rule matching
                
            # Check domain names
            for domain_pattern in info['domains']:
                if domain_pattern in domain_name:
                    rule_boosts[category] += 0.3
            
            # Check TLDs
            for tld_pattern in info['tlds']:
                if tld_pattern.lstrip('.') == tld:
                    rule_boosts[category] += 0.2
            
            # Check keywords in text
            keyword_matches = sum(1 for keyword in info['keywords'] if keyword in text_lower)
            if keyword_matches > 0:
                rule_boosts[category] += min(0.2, keyword_matches * 0.05)
        
        # Combine ensemble predictions with rules
        category_scores = defaultdict(float)
        
        # Add ensemble scores (exclude "Other" predictions)
        for pred, conf in zip(predictions, confidences):
            if pred != "Άλλο":
                category_scores[pred] += conf
        
        # Apply rule boosts
        for category, boost in rule_boosts.items():
            if category != "Άλλο":
                category_scores[category] += boost
        
        # Get final prediction
        if not category_scores:
            return "Άλλο", 0.1
        
        best_category_name = max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence as average of only the scores that predicted this category
        matching_confidences = []
        for pred, conf in zip(predictions, confidences):
            if pred == best_category_name:
                matching_confidences.append(conf)
        
        # If no direct matches (only rule-based), use a base confidence
        if not matching_confidences:
            final_confidence = 0.5
        else:
            final_confidence = sum(matching_confidences) / len(matching_confidences)
        
        return best_category_name, min(1.0, final_confidence)
    

    
    def process_data(self, df, batch_size=500, output_file="classified_domains.parquet"):
        """Process data with enhanced classification"""
        print(f"🔄 Processing {len(df)} domains in batches of {batch_size}...")
        
        # Check for existing results to resume from
        start_batch = 0
        existing_results = []
        
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_parquet(output_file)
                existing_count = len(existing_df)
                start_batch = existing_count // batch_size
                existing_results = existing_df.to_dict('records')
                
                print(f"📋 Found existing results: {existing_count} domains already processed")
                print(f"🔄 Resuming from batch {start_batch + 1} (starting at domain {existing_count + 1})")
                
                # Skip already processed domains
                df = df.iloc[existing_count:].reset_index(drop=True)
                
            except Exception as e:
                print(f"⚠️  Could not read existing results: {e}")
                print(f"🔄 Starting fresh...")
                start_batch = 0
                existing_results = []
        
        if len(df) == 0:
            print("✅ All domains already processed!")
            return pd.read_parquet(output_file) if os.path.exists(output_file) else pd.DataFrame()
        
        results = existing_results.copy()
        all_texts = []
        
        # Calculate total number of batches for remaining data
        total_batches = (len(df) + batch_size - 1) // batch_size
        total_batches_overall = start_batch + total_batches
        
        # Process in batches with overall progress bar
        batch_progress = tqdm(range(total_batches), desc="Processing batches", unit="batch")
        
        for batch_num in batch_progress:
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            batch_results = []
            batch_texts = []
            
            # Calculate actual batch number (including already processed)
            actual_batch_num = start_batch + batch_num + 1
            actual_domain_start = len(existing_results) + start_idx + 1
            actual_domain_end = len(existing_results) + end_idx
            
            # Update batch progress description
            batch_progress.set_description(f"Batch {actual_batch_num}/{total_batches_overall} (domains {actual_domain_start}-{actual_domain_end})")
            
            # Progress bar for domains within this batch
            domain_desc = f"  └─ Classifying domains"
            
            for idx, row in tqdm(batch.iterrows(), total=len(batch), desc=domain_desc, leave=False, position=1):
                # Preprocess
                text, domain_features = self.preprocess_text(row)
                
                # Classify
                category, confidence = self.classify_with_ensemble(text, domain_features)
                
                # Convert Greek category to English for output
                english_category = self._convert_to_english(category)
                
                # Store result - preserve all original columns and add new ones
                result = row.to_dict()  # Keep all original columns
                result['category'] = english_category  # Add category
                result['confidence'] = confidence      # Add confidence
                
                batch_results.append(result)
                if text:
                    batch_texts.append(text)
                    all_texts.append(text)
            
            results.extend(batch_results)
            
            # Save progress after each batch (incremental save)
            try:
                results_df = pd.DataFrame(results)
                results_df.to_parquet(output_file, index=False)
            except Exception as e:
                tqdm.write(f"⚠️  Warning: Could not save progress: {e}")
            
            # Update batch progress with completion info
            batch_progress.set_postfix({
                'domains': f"{len(batch_results)}",
                'total_processed': f"{len(results)}"
            })
            
            # Memory cleanup every 5 batches (2500 domains)
            if actual_batch_num % 5 == 0:
                gc.collect()
                tqdm.write(f"🧹 Memory cleanup after {actual_batch_num} batches")
        
        batch_progress.close()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        

        
        # Save results
        results_df.to_parquet(output_file, index=False)
        print(f"✅ Results saved to {output_file}")
        
        # Print summary statistics
        self._print_summary(results_df)
        
        return results_df
    
    def _print_summary(self, results_df):
        """Print classification summary"""
        print("\n📈 Classification Summary:")
        print("=" * 50)
        
        category_counts = results_df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(results_df)) * 100
            avg_confidence = results_df[results_df['category'] == category]['confidence'].mean()
            conflict_marker = " (CONFLICT RESOLVED)" if category.startswith('*') else ""
            print(f"{category:30s}: {count:6d} ({percentage:5.1f}%) - Avg Confidence: {avg_confidence:.3f}{conflict_marker}")
        
        print(f"\nTotal domains classified: {len(results_df):,}")
        print(f"Average confidence: {results_df['confidence'].mean():.3f}")
        print(f"High confidence (>0.7): {len(results_df[results_df['confidence'] > 0.7]):,} ({len(results_df[results_df['confidence'] > 0.7])/len(results_df)*100:.1f}%)")
        
        # Additional statistics
        conflicts = len(results_df[results_df['category'].str.startswith('*')])
        vague = len(results_df[results_df['category'] == 'Other'])
        
        print(f"\n🔄 Conflict Resolution Statistics:")
        print(f"Domains with conflicts (marked with *): {conflicts:,} ({conflicts/len(results_df)*100:.1f}%)")
        print(f"Vague classifications (Other): {vague:,} ({vague/len(results_df)*100:.1f}%)")
        print(f"Successfully classified: {len(results_df) - vague:,} ({(len(results_df) - vague)/len(results_df)*100:.1f}%)")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python enhanced_domain_classifier.py <oscar_metadata.parquet>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found!")
        sys.exit(1)
    
    print("🚀 Enhanced Domain Classification Starting...")
    print("=" * 60)
    
    # Load data
    print(f"📖 Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"   Loaded {len(df):,} domains")
    
    # Initialize classifier
    classifier = EnhancedDomainClassifier(use_gpu=True)
    
    # Setup models
    classifier.setup_models()
    
    # Process data
    output_file = input_file.replace('.parquet', '_enhanced_classified.parquet')
    results_df = classifier.process_data(df, output_file=output_file)
    
    print("\n🎉 Enhanced classification completed!")
    print(f"📊 Results saved to: {output_file}")


if __name__ == "__main__":
    main() 