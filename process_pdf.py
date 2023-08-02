import json
import replicate
import PyPDF2


def extract_images_from_pdf(pdf_path):
    image_paths = []
    meta_data = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            for idx, image_file_object in enumerate(page.images):
                with open(image_file_object.name, "wb") as fp:
                    fp.write(image_file_object.data)
                image_paths.append(image_file_object.name)
                meta_data.append({
                    "name": image_file_object.name,
                    "page_num": page_num,
                    "position_in_page": idx,
                })

    return image_paths, meta_data


def generate_caption(image_path):
    output = replicate.run(
        "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
        input={"image": open(image_path, "rb")}
    )
    return output


def save_json_file(file_path, data):
    json_string = json.dumps(data, indent=4)
    with open(file_path, "w") as json_file:
        json_file.write(json_string)


def main(pdf_path):
    image_paths, meta_data = extract_images_from_pdf(pdf_path)
    for idx, image_path in enumerate(image_paths):
        caption = generate_caption(image_path)
        meta_data[idx]["caption"] = caption
        print(f"Caption for Image {idx}: {caption}")

    save_json_file("meta_data.json", meta_data)


if __name__ == "__main__":
    main("LP-FT.pdf")
