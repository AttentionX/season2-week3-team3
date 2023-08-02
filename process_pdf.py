import json
import replicate
import pypdf


def extract_text_from_pdf(pdf_path):
    image_paths = []
    image_meta_data = []
    text_per_page = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text_per_page.append(text)
            for idx, image_file_object in enumerate(page.images):
                with open(image_file_object.name, "wb") as fp:
                    fp.write(image_file_object.data)
                image_paths.append(image_file_object.name)
                image_meta_data.append({
                    "name": image_file_object.name,
                    "page_num": page_num,
                    "position_in_page": idx,
                })

    return text_per_page, image_paths, image_meta_data


def generate_caption(image_path):
    output = replicate.run(
        "gfodor/instructblip:ca869b56b2a3b1cdf591c353deb3fa1a94b9c35fde477ef6ca1d248af56f9c84",
        input={
            "image_path": open(image_path, "rb"),
            "prompt": "describe the figure image in the deep learning paper.",
        }
    )
    return output


def save_json_file(file_path, data):
    json_string = json.dumps(data, indent=4)
    with open(file_path, "w") as json_file:
        json_file.write(json_string)


def main(pdf_path):
    text_per_page, image_paths, image_meta_data = extract_text_from_pdf(pdf_path)
    for idx, image_path in enumerate(image_paths):
        caption = generate_caption(image_path)
        image_meta_data[idx]["caption"] = caption
        print("-"*20)
        print(f"Caption for Image {idx}: {caption}")

        page_num = image_meta_data[idx]["page_num"]
        text_per_page[page_num] += ("\nFigure. " + caption + "\n")
    
    save_json_file("text_per_page.json", text_per_page)
    save_json_file("image_meta_data.json", image_meta_data)

    print("Done!!!")


if __name__ == "__main__":
    main("LP-FT.pdf")
