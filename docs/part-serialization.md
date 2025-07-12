# "Part" // serialization features 

I want to be able to have support for common formats:

* Audio
* Image
* Video
* Text
* PDF

I want to be able to load from multiple sources:

* "Raw"
* File
* Gdrive

I want the internal representation to be "native"

As in, if I have an image I want to represent it with PIL, if I have an audio
clip I want it to be represented with librosa etc.

## Representation

We can get the first part with our "Part" representation:

{ mime_type, data }

We can get the second type by registering a URL loader:

scheme:///...

For the third, I think we just need to defer to the appropriate loader
when resolving from a dictionary etc:

DataType.from_dict(...)

Then I can have AudioPart be a chunk with support for all sorts of extra
functionality.