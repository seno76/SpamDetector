"""
Создание отдельного тестового датасета (unseen data)
"""
from pathlib import Path

def create_test_natural_texts():
    """Создание тестовых обычных текстов"""
    test_natural = [
        "Artificial intelligence has transformed many industries in recent years. Machine learning algorithms can now perform tasks that previously required human intelligence.",
        
        "The solar system consists of the Sun and everything that orbits around it. This includes eight planets, numerous moons, asteroids, and comets.",
        
        "Photosynthesis is the process by which plants convert light energy into chemical energy. This fundamental biological process sustains most life on Earth.",
        
        "Democracy is a form of government where citizens exercise power by voting. Free and fair elections are essential components of democratic systems.",
        
        "The Renaissance was a period of cultural rebirth in Europe. It marked significant developments in art, literature, science, and philosophy.",
        
        "Quantum mechanics describes the behavior of matter at atomic and subatomic scales. This theory fundamentally changed our understanding of physics.",
        
        "Biodiversity refers to the variety of life forms on Earth. Protecting ecosystems is crucial for maintaining this biological diversity.",
        
        "Economics studies how societies allocate scarce resources. Supply and demand are fundamental concepts in economic theory.",
        
        "The human brain contains approximately 86 billion neurons. These cells communicate through complex networks of synaptic connections.",
        
        "Literature encompasses written works of artistic merit. Poetry, novels, and plays are major literary forms across cultures.",
        
        "Archaeology studies human history through excavation and analysis of artifacts. This field provides insights into ancient civilizations.",
        
        "Meteorology is the science of atmospheric phenomena. Weather forecasting relies on complex mathematical models and observations.",
        
        "Genetics explores heredity and variation in living organisms. DNA sequences contain instructions for biological development.",
        
        "Philosophy examines fundamental questions about existence and knowledge. Ethics, logic, and metaphysics are major branches of philosophy.",
        
        "Computer networks enable communication between devices worldwide. The Internet protocol suite facilitates global data transmission.",
        
        "Astronomy studies celestial objects and phenomena beyond Earth. Telescopes allow observation of distant galaxies and stellar systems.",
        
        "Chemistry investigates the composition and properties of matter. Chemical reactions involve transformation of substances at molecular levels.",
        
        "Sociology analyzes human social behavior and institutions. Cultural norms and social structures shape individual and group interactions.",
        
        "Mathematics provides formal frameworks for logical reasoning. Abstract concepts in mathematics have practical applications across sciences.",
        
        "Renewable energy technologies harness natural processes for power generation. Solar panels and wind turbines convert environmental energy into electricity."
    ]
    
    return test_natural

def create_test_spam_texts():
    """Создание тестовых спам текстов"""
    test_spam = [
        "BUY NOW!!! LIMITED OFFER!!! Get 90% discount on all products today only! Click here immediately! FREE SHIPPING worldwide! Don't miss this amazing deal!!!",
        
        "CONGRATULATIONS! You won $1,000,000! Claim your prize now! Click link below! Hurry offer expires soon! Winner winner winner! Cash prize waiting!!!",
        
        "Lose weight FAST! Miracle pills! Guaranteed results in 7 days! Buy cheap online! Best price! Doctor recommended! Order now get bonus! Free trial!!!",
        
        "Work from HOME! Earn $5000 weekly! Easy money! No experience needed! Join now! Financial freedom! Make money online fast! Limited spots available!!!",
        
        "CASINO BONUS! Free spins! Jackpot slots! Win big money! Gambling paradise! Poker blackjack roulette! Best casino! Deposit now get 200% bonus!!!",
        
        "Viagra cialis online! Cheap pharmacy! Discount pills! Best price guaranteed! Medications online! Buy now! Prescription drugs! Fast delivery worldwide!!!",
        
        "HOT SINGLES near you! Dating site! Meet tonight! Beautiful women! Chat now! Find love! Romance dating! Adult entertainment! Click to register free!!!",
        
        "CREDIT APPROVED! Instant loans! Bad credit OK! Cash advance! Money today! Apply now! Guaranteed approval! Low interest! Payday loans fast!!!",
        
        "SEO services! First page Google! Backlinks guaranteed! Website traffic! Ranking optimization! Best SEO! Cheap prices! Results guaranteed! Order today!!!",
        
        "iPhone MacBook FREE! Contest winner! Claim gadget now! Electronics giveaway! Latest models! Apple products! Limited quantity! Enter code! Hurry!!!",
        
        "REFINANCE mortgage! Lower rates! Save thousands! Home loans! Debt consolidation! Apply online! Approved in minutes! Bad credit welcome! Act now!!!",
        
        "Make MILLIONS! Investment opportunity! Stock tips! Crypto trading! Financial advisor! Rich quick! Trading secrets! Profit guaranteed! Join now!!!",
        
        "Male enhancement! Performance pills! Natural formula! Results guaranteed! Bigger stronger! Order discreet! Doctor approved! Try risk free! Buy today!!!",
        
        "Online degree! University diploma! Accredited programs! Fast graduation! Cheap tuition! Study online! Certificate programs! Enroll now! No exams!!!",
        
        "FORECLOSURE alert! Bank account compromised! Verify identity! Click immediately! Urgent action required! Security warning! Update information! Suspended!!!",
        
        "Tax refund waiting! IRS notification! Claim money! Government grant! Financial assistance! Free money! Stimulus check! Process immediately! Deadline!!!",
        
        "Luxury watches! Replica designer! Rolex Omega! Cheap authentic! Swiss movement! Buy online! Free shipping! Best quality! Limited stock! Order!!!",
        
        "Anti-aging miracle! Wrinkle cream! Look younger! Celebrity secret! Remove years! Beauty treatment! Natural ingredients! Money back guarantee! Try free!!!",
        
        "Tech support! Computer virus detected! Call now! System warning! Security breach! Microsoft certified! Remote assistance! Fix immediately! Toll free!!!",
        
        "EARN passive income! Cryptocurrency mining! Bitcoin profits! Investment platform! Trading robot! Automated system! No work required! Join thousands! Rich!!!"
    ]
    
    return test_spam

def save_test_dataset():
    """Сохранение тестового датасета"""
    # Создаем папки
    test_dir = Path('data/test')
    (test_dir / 'natural').mkdir(parents=True, exist_ok=True)
    (test_dir / 'spam').mkdir(parents=True, exist_ok=True)
    
    # Сохраняем natural
    natural_texts = create_test_natural_texts()
    for i, text in enumerate(natural_texts):
        with open(test_dir / 'natural' / f'test_natural_{i}.txt', 'w', encoding='utf-8') as f:
            f.write(text)
    
    # Сохраняем spam
    spam_texts = create_test_spam_texts()
    for i, text in enumerate(spam_texts):
        with open(test_dir / 'spam' / f'test_spam_{i}.txt', 'w', encoding='utf-8') as f:
            f.write(text)
    
    print("✅ ТЕСТОВЫЙ ДАТАСЕТ СОЗДАН!")
    print(f"📂 data/test/natural/ → {len(natural_texts)} файлов")
    print(f"📂 data/test/spam/ → {len(spam_texts)} файлов")
    print(f"📊 ВСЕГО: {len(natural_texts) + len(spam_texts)} тестовых текстов")
    
    return len(natural_texts), len(spam_texts)

if __name__ == "__main__":
    save_test_dataset()
