# Report

## Neural Machine Translation model 

#### Seq2seq

- Trong mô hình seq2seq dùng cho bài toán NMT (Neural Machine Translation) bao gồm 2 mạng RNN chính: Encoder và Decoder. Encoder với đầu vào là câu ở ngôn ngữ gốc, đầu ra tại layer cuối cùng của Encoder gọi là 1 context vector. Với ý nghĩa lượng thông tin từ câu của Encoder sẽ được tóm gọn lại trong 1 vector đầu ra cuối cùng. Từ đó, Decoder dùng chính context vector đó, cùng với hidden state và từ trước đó để predict từ tiếp theo tại decoder qua từng timestep.

![encode-decode arch img](https://github.com/tensorflow/nmt/raw/master/nmt/g3doc/img/encdec.jpg)

![NMT arch](https://github.com/tensorflow/nmt/raw/master/nmt/g3doc/img/seq2seq.jpg)

#### Attention

  > Bahdanau: ...we conjecture that the use of a fixed-length vector is a
bottleneck in improving the performance of this basic encoder–decoder architecture,
and propose to extend this by allowing a model to automatically (soft-)search
for parts of a source sentence that are relevant to predicting a target word, without
having to form these parts as a hard segment explicitly

- Việc encode toàn bộ thông tin từ source vào 1 vector cố định khiến việc mô hình khi thực hiện trên các câu dài (long sentence) không thực sự tốt, mặc dù sử dụng LSTM (BiLSTM, GRU) để khắc phục điểm yếu của mạng RNN truyền thống với hiện tượng Vanishing Gradient, nhưng như thế có vẻ vẫn chưa đủ, đặc biệt đối với những câu dài hơn những câu trong training data. Từ đó, trong paper, tác giả Bahdanau đề xuất 1 cơ chế cho phép mô hình có thể chú trọng vào những phần quan trọng (word liên kết với word từ source đến target), và thay vì chỉ sử dụng context layer được tạo ra từ layer cuối cùng của Encoder, tác giả sử dụng tất cả các output của từng cell qua từng timestep, kết hợp với hidden state của từng cell để "tổng hợp" ra 1 context vector (attention vector) và dùng nó làm đầu vào cho từng cell trong Decoder. 

*Cơ chế "tổng hợp" Attention trong paper của tác giả Bahdanau*: Align and Jointly model (Additive Attention).

![Bahdanau - Align and Jointly model](https://lilianweng.github.io/lil-log/assets/images/encoder-decoder-attention.png)

- Ma trận bất đối xứng (confusion matrix) được tạo ra bởi alignment score, thể hiện mức độ tương quan correlation giữa source và target.
![](https://lilianweng.github.io/lil-log/assets/images/bahdanau-fig3.png)


#### Optimizer (SGD, Adam)

#### Inference mode (Beam search, Greedy)

## Measure performance with BLEU score

## Training with different hyperparameters

- IWSLT English-Vietnamese dataset
- Data Input Pipeline

### Impact of such hyperparameters

## Result with Attention Matrix
