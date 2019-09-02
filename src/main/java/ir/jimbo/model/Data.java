package ir.jimbo.model;

import lombok.*;

import java.io.Serializable;

@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class Data implements Serializable {
    double tag;
    String content;
}
