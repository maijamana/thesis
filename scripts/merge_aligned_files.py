"""
Скрипт для об'єднання line-aligned файлів simple.aligned та normal.aligned
в один чистий датасет з парами (original, simplified).
"""
import pandas as pd
from pathlib import Path
import argparse
from typing import Tuple, Optional


def parse_line(line: str) -> Optional[Tuple[str, int, str]]:
    """
    Парсить рядок формату: topic\tindex\tsentence
    
    Returns:
        (topic, index, sentence) або None якщо рядок некоректний
    """
    line = line.strip()
    if not line:
        return None
    
    parts = line.split('\t', 2)
    if len(parts) != 3:
        return None
    
    topic = parts[0].strip()
    try:
        index = int(parts[1].strip())
    except ValueError:
        return None
    
    sentence = parts[2].strip()
    if not sentence:
        return None
    
    return (topic, index, sentence)


def clean_sentence(sentence: str) -> str:
    """Очищає речення від зайвих пробілів та спецсимволів."""
    return ' '.join(sentence.split())


def is_valid_pair(original: str, simplified: str) -> bool:
    """
    Перевіряє, чи пара речень валідна (не дублікат, не порожня, має сенс).
    """
    if not original or not simplified:
        return False
    
    # Видаляємо дублікати (якщо оригінал і спрощене однакові)
    if original.strip() == simplified.strip():
        return False
    
    # Мінімальна довжина
    if len(original.strip()) < 5 or len(simplified.strip()) < 5:
        return False
    
    return True


def merge_aligned_files(
    simple_path: Path,
    normal_path: Path,
    output_path: Path,
    min_length: int = 5,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Об'єднує line-aligned файли в один датасет.
    
    Args:
        simple_path: Шлях до simple.aligned (спрощені)
        normal_path: Шлях до normal.aligned (оригінальні)
        output_path: Шлях для збереження CSV
        min_length: Мінімальна довжина речення
        remove_duplicates: Видаляти дублікати
    
    Returns:
        DataFrame з колонками: topic, index_simple, index_normal, original, simplified
    """
    print(f"Читаємо файли...")
    print(f"  Simple: {simple_path}")
    print(f"  Normal: {normal_path}")
    
    pairs = []
    skipped = 0
    
    with open(simple_path, 'r', encoding='utf-8') as f_simple, \
         open(normal_path, 'r', encoding='utf-8') as f_normal:
        
        line_num = 0
        for line_simple, line_normal in zip(f_simple, f_normal):
            line_num += 1
            
            # Парсимо рядки
            simple_data = parse_line(line_simple)
            normal_data = parse_line(line_normal)
            
            if not simple_data or not normal_data:
                skipped += 1
                continue
            
            topic_simple, index_simple, sentence_simple = simple_data
            topic_normal, index_normal, sentence_normal = normal_data
            
            # Очищаємо речення
            simplified = clean_sentence(sentence_simple)
            original = clean_sentence(sentence_normal)
            
            # Перевіряємо валідність пари
            if not is_valid_pair(original, simplified):
                skipped += 1
                continue
            
            # Перевіряємо мінімальну довжину
            if len(original) < min_length or len(simplified) < min_length:
                skipped += 1
                continue
            
            pairs.append({
                'topic': topic_normal,  # Використовуємо topic з normal
                'index_normal': index_normal,
                'index_simple': index_simple,
                'original': original,
                'simplified': simplified
            })
            
            if line_num % 10000 == 0:
                print(f"  Оброблено {line_num} рядків, знайдено {len(pairs)} пар...")
    
    print(f"\nВсього оброблено {line_num} рядків")
    print(f"Пропущено {skipped} невалідних рядків")
    print(f"Знайдено {len(pairs)} валідних пар")
    
    # Створюємо DataFrame
    df = pd.DataFrame(pairs)
    
    if df.empty:
        print("⚠️  Помилка: не знайдено жодної валідної пари!")
        return df
    
    # Видаляємо дублікати за оригінальним реченням
    if remove_duplicates:
        initial_count = len(df)
        df = df.drop_duplicates(subset=['original'], keep='first')
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            print(f"Видалено {duplicates_removed} дублікатів за оригінальним реченням")
    
    # Сортуємо за topic та index_normal
    df = df.sort_values(['topic', 'index_normal']).reset_index(drop=True)
    
    print(f"\nФінальний датасет: {len(df)} пар")
    print(f"Унікальних тем: {df['topic'].nunique()}")
    
    # Зберігаємо
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Результати збережено у: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Об'єднує line-aligned файли simple.aligned та normal.aligned в один датасет"
    )
    parser.add_argument(
        '--simple',
        type=Path,
        default=Path('data/simple.aligned'),
        help='Шлях до simple.aligned (спрощені речення)'
    )
    parser.add_argument(
        '--normal',
        type=Path,
        default=Path('data/normal.aligned'),
        help='Шлях до normal.aligned (оригінальні речення)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/aligned_pairs.csv'),
        help='Шлях для збереження CSV (default: data/aligned_pairs.csv)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=5,
        help='Мінімальна довжина речення (default: 5)'
    )
    parser.add_argument(
        '--keep-duplicates',
        action='store_true',
        help='Не видаляти дублікати за оригінальним реченням'
    )
    
    args = parser.parse_args()
    
    if not args.simple.exists():
        print(f"❌ Помилка: файл {args.simple} не знайдено")
        return
    
    if not args.normal.exists():
        print(f"❌ Помилка: файл {args.normal} не знайдено")
        return
    
    merge_aligned_files(
        simple_path=args.simple,
        normal_path=args.normal,
        output_path=args.output,
        min_length=args.min_length,
        remove_duplicates=not args.keep_duplicates
    )


if __name__ == "__main__":
    main()
